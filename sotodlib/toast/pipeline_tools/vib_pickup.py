import numpy as np

from toast.timing import Timer

from ..vib_pickup import OpVibPickup


def add_vib_pickup_args(parser):
    parser.add_argument(
        "--add_vib_pickup",
        required=False,
        action="store_true"
    )
    parser.add_argument(
        "--vib-freq",
        required=False,
        type=np.float,
        help="Frequency of vibrational pickup in Hz",
    )
    parser.add_argument(
        "--vib-amp-mean",
        type=np.float,
        help="Amplitude of vibrational pickup in [Units]",
    )
    parser.add_argument(
        "--vib-amp-std",
        type=np.float,
        help="Standard deviation of the amplitude distribution",
    )
    return


def add_vib_pickup(args, comm, data, name):
    if not args.add_vib_pickup:
        return
    timer = Timer()
    timer.start()
    vibop = OpVibPickup(name=name,
                        vib_freq=args.vib_freq,
                        vib_amp_mean=args.vib_amp_mean,
                        vib_amp_std=args.vib_amp_std)
    vibop.exec(data)
    timer.report_clear("Add Vibrational Pickup")
