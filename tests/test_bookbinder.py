try:
    import socs.agents.smurf_file_emulator.agent as sfe
except ModuleNotFoundError:
    sfe = None
import sotodlib.io.bookbinder as bb
import numpy as np
import so3g
from spt3g import core
import os
from unittest import mock
import unittest
import tempfile

def load_data(files, data_name='data'):
    """
    Load in G3 data and timestamps
    """
    it = bb.get_frame_iter(files)
    data = []
    times = []
    for frame in it:
        if frame.type != core.G3FrameType.Scan:
            continue
        data.append(frame[data_name].data)
        times.append(np.array(frame[data_name].times) / core.G3Units.s)
    data = np.hstack(data)
    times = np.hstack(times)
    return times, data

class BookbinderTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        print("tempdir: {}".format(self.tempdir.name))
        self.l2_dir = os.path.join(self.tempdir.name, 'l2_archive')
        self.l3_dir = os.path.join(self.tempdir.name, 'books')
        os.makedirs(self.l2_dir, exist_ok=True)
        os.makedirs(self.l3_dir, exist_ok=True)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_data(self, stream_id, duration, sample_rate, nchans,
                   drop_chance=0.1, start_offset=0):
        """
        Writes realistic smurf data using the smurf file emulator
        """
        frame_len = 5
        file_duration = 60*10
        mock_agent = mock.MagicMock()
        os.makedirs(self.l2_dir, exist_ok=True)

        parser = sfe.make_parser()
        args = parser.parse_args(args=[
            '--stream-id', stream_id, 
            '--base-dir', self.l2_dir,
            '--file-duration', str(file_duration),
            '--frame-len', str(frame_len),
            '--sample-rate', str(sample_rate),
            '--drop-chance', str(drop_chance),
            '--nchans', str(nchans), ])
        em = sfe.SmurfFileEmulator(mock_agent, args)
        session = mock.MagicMock()
        session.data = {}
        em.uxm_setup(session, {'sleep': False})
        em.stream(session, dict(
            duration=duration, use_stream_between=True, start_offset=start_offset
        ))
        return session.data['g3_files']
    
    def test_smurf_stream_processor(self):
        if sfe is None:
            print(
                "No socs module. "
                "Skipping BookbinderTest.test_smurf_stream_processor"
            )
            return
        l2_files = self.write_data('test', 60, 200, 5, drop_chance=0.1)

        in_ts, in_data = load_data(l2_files, data_name='data')

        ts, mask = bb.fill_time_gaps(in_ts)
        frame_idxs = np.zeros_like(ts, dtype=np.int32)
        file_idxs = np.zeros_like(np.unique(frame_idxs), dtype=np.int32)

        obsid = 'obsid'
        bookid = 'bookid'
        readout_ids = [str(i) for i in range(len(in_data))]
        ssp = bb.SmurfStreamProcessor(
            obsid, l2_files, bookid, readout_ids, allow_bad_timing=False
        )
        ssp.preprocess()
        ssp.bind(self.l3_dir, ts, frame_idxs, file_idxs)

        out_ts, out_data = load_data(ssp.out_files, data_name='signal')

        self.assertTrue(np.all(out_ts[mask] == in_ts))
        self.assertTrue(np.all(out_data[:, mask] == in_data))

if __name__ == '__main__':
    unittest.main()
