import os
import argparse
import numpy as np

import datetime as dt
from collections import OrderedDict

from sqlalchemy import or_, and_, not_

from sotodlib.io.load_smurf import G3tSmurf, Observations

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_prefix', 
                        help="The prefix to data locations, use individual"
                        " locations flags if data is stored in non-standard folders")
    parser.add_argument('db_path', help="Path to database to search")
    parser.add_argument('--min-ctime', help="Minimum ctime to look for overlaps")
    parser.add_argument('--max-ctime', help="Maximum ctime to look for overlaps")
    parser.add_argument('--stream-ids',help="A comma separated list of"
                        " stream_ids to use to look for overlapping"
                        " observations")
    parser.add_argument('--min-overlap', type=float, default=0,
                        help="Minimum overlap (in seconds) required to be an"
                        " overlapping observation")    
    parser.add_argument("--ignore-singles", action="store_true",
                        help="If passed, script will not return observations with"
                        " no overlapping segments")
    parser.add_argument('--timestream-folder', 
                        help="Absolute path to folder with .g3 timestreams."
                        " Overrides data_prefix")
    parser.add_argument('--smurf-folder', 
                        help="Absolute path to folder with pysmurf archived data."
                        " Overrides data_prefix")
    
    return parser

def main(args=None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()
    
    if args.timestream_folder is None:
        args.timestream_folder = os.path.join( args.data_prefix, 'timestreams/')
       
    if args.smurf_folder is None:
        args.smurf_folder = os.path.join( args.data_prefix, 'smurf/')
        
    SMURF = G3tSmurf(args.timestream_folder, 
                     db_path=args.db_path,
                     meta_path=args.smurf_folder)
    session = SMURF.Session()
    
    if args.min_ctime is None:
        args.min_ctime = session.query(Observations.timestamp).order_by(Observations.timestamp).first()[0]
    if args.max_ctime is None:
        args.max_ctime = dt.datetime.now().timestamp()
            

    ## find all complete observations that start within the time range
    obs_q = session.query(Observations).filter(Observations.timestamp >= args.min_ctime,
                                               Observations.timestamp <= args.max_ctime,
                                           not_(Observations.stop==None))

    if args.stream_ids is None:
        ## find all unique stream ids during the time range
        streams = session.query(Observations.stream_id).filter(
                          Observations.timestamp >= args.min_ctime,
                          Observations.timestamp <= args.max_ctime
                          ).distinct().all()
        streams = [s[0] for s in streams]
    else:
        # format like the query response
        streams = args.stream_ids.split(',')
        streams = [s.strip() for s in streams]

    stream_filt = or_(*[Observations.stream_id == s for s in streams])

    output = []
    for stream in streams:

        ## loop through all observations for this particular stream_id
        for str_obs in obs_q.filter(Observations.stream_id == stream).all(): 
            ## query for all possible types of overlapping observations
            q = obs_q.filter(
                Observations.stream_id != str_obs.stream_id,
                stream_filt,
                or_(  
                    and_(Observations.start <= str_obs.start, Observations.stop >= str_obs.start),
                    and_(Observations.start <= str_obs.stop, Observations.start >= str_obs.start),
                    and_(Observations.start >= str_obs.start, Observations.stop <= str_obs.stop),
                    and_(Observations.start <= str_obs.start, Observations.stop >= str_obs.stop)
                ))
            if q.count()==0 and not args.ignore_singles:
                # observation has no overlapping segments    
                output.append( (str_obs.obs_id,) )

            elif q.count() > 0:
                obs_list = q.all()
                obs_list.append(str_obs)

                ## check to make sure ALL observations overlap all others
                if np.max([o.start for o in obs_list]) > np.min([o.stop for o in obs_list]):
                    continue
                
                overlap_time = np.min([o.stop for o in obs_list]) - np.max([o.start for o in obs_list])
                if overlap_time.total_seconds() < args.min_overlap:
                    continue
                    
                ## add all of the possible overlaps
                id_list = [obs.obs_id for obs in obs_list]                   
                output.append( tuple(sorted(id_list)) )

    ## remove exact duplicates
    output = list(OrderedDict.fromkeys(output))
    return output

if __name__ == '__main__':
    print(main())
