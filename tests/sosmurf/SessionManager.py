from spt3g import core
import yaml
import time
from enum import Enum


SOSTREAM_VERSION = 1


class FlowControl(Enum):
    """Flow control enumeration."""
    ALIVE = 0
    START = 1
    END = 2
    CLEANSE = 3


class SessionManager:
    enable_streams = "AMCc.SmurfProcessor.FileWriter.IsOpen"

    def __init__(self, stream_id=''):
        self.stream_id = stream_id
        self.session_id = None
        self.end_session_flag = False
        self.frame_num = 0
        self.status = {}

    def flowcontrol_frame(self, fc):
        """
        Creates flow control frame.

        Args:
            fc (int):
                flow control type
        """
        frame = core.G3Frame(core.G3FrameType.none)
        frame['sostream_flowcontrol'] = fc.value
        return frame

    def tag_frame(self, frame):
        frame['sostream_version'] = SOSTREAM_VERSION
        frame['sostream_id'] = self.stream_id
        frame['frame_num'] = self.frame_num
        self.frame_num += 1
        if self.session_id is not None:
            frame['session_id'] = self.session_id
        if 'time' not in frame:
            frame['time'] = core.G3Time.Now()

        return frame

    def status_frame(self):
        frame = core.G3Frame(core.G3FrameType.Wiring)
        frame['status'] = yaml.safe_dump(self.status)
        frame['dump'] = 1

        self.tag_frame(frame)
        return frame

    def start_session(self):
        self.session_id = int(time.time())

        frame = core.G3Frame(core.G3FrameType.Observation)

        self.tag_frame(frame)
        return frame

    def __call__(self, frame):
        out = [frame]

        #######################################
        # On None frames
        #######################################
        if frame.type == core.G3FrameType.none:

            if self.end_session_flag:
                # Returns [previous, end, obs cleanse, wiring cleanse]
                out = []
                out.append(self.flowcontrol_frame(FlowControl.END))

                f = core.G3Frame(core.G3FrameType.Observation)
                f['sostream_flowcontrol'] = FlowControl.CLEANSE.value
                out.append(f)

                f = core.G3Frame(core.G3FrameType.Wiring)
                f['sostream_flowcontrol'] = FlowControl.CLEANSE.value
                out.append(f)

                self.session_id = None
                self.end_session_flag = False
                self.frame_num = 0
                return out

        #######################################
        # On Scan frames
        #######################################
        elif frame.type == core.G3FrameType.Scan:

            if self.session_id is None:
                return []

            self.tag_frame(frame)
            return out

        #######################################
        # On Wiring frames
        #######################################
        elif frame.type == core.G3FrameType.Wiring:

            status_update = yaml.safe_load(frame['status'])

            # Get difference between status-frame and current status.
            diff = {}
            for k, v in status_update.items():
                if k not in self.status:
                    diff[k] = v
                elif v != self.status[k]:
                    diff[k] = v
            if len(diff) == 0:  # Skip if there's no difference
                return []

            self.status.update(status_update)

            # Replace full status update with difference
            del frame['status']
            frame['status'] = yaml.dump(diff)

            enable = int(status_update.get(self.enable_streams, -1))
            if self.session_id is None:
                if enable == 1:
                    # Returns [start, session, status]
                    session_frame = self.start_session()
                    out = [
                        self.flowcontrol_frame(FlowControl.START),
                        session_frame,
                        self.status_frame()
                    ]

                    return out
                else:
                    # Don't output any status frames if session is not active
                    return []
            else:
                frame['dump'] = 0
                if enable == 0:
                    self.end_session_flag = True
                self.tag_frame(frame)
                return out
