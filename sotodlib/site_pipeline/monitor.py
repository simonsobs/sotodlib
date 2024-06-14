import yaml

from influxdb import InfluxDBClient

class Monitor:
    def __init__(self, host, port, database='qds', username=u'root', password=u'root', path='', ssl=False):
        """QDS Monitor, an interface to monitoring data quality in InfluxDB.

        Parameters
        ----------
        host : str
            InfluxDB host address
        port : int
            InfluxDB port number
        database : str
            InfluxDB database. Will be created if it does not exist already.
            Defaults to 'qds'.
        username : str
            Username for the InfluxDB, defaults to 'root'.
        password : str
            Password for the InfluxDB, defaults to 'root'.
        path : str
            Path of InfluxDB on the server to connect to, defaults to ''
        ssl : bool
            Use https to connect, defaults to False

        Attributes
        ----------
        client : influxdb.client.InfluxDBClient
            InfluxDB client
        queue : list
            InfluxQL line formatted entries for upload to InfluxDB. Recorded
            entries are "queued" to this list and written with Monitor.write().

        """
        self.client = Monitor._connect_to_db(host, port, database, username, password, path, ssl)
        self.queue = []

    @classmethod
    def from_configs(cls, configs):
        """Create a monitor from a configuration file
        
        Parameters
        ----------
        configs: dict or string
            configuration dictionary or string that's a file name that can be
            loaded by yaml into a configuration dictionary

        Returns
        -------
        connected Monitor Instance
        """
        if type(configs) == str:
            configs = yaml.safe_load( open(configs, "r") )    
        return cls(
            host = configs["host"],
            port = configs["port"],
            database = configs["database"],
            username = configs["username"],
            password = configs["password"],
            path = configs["path"],
            ssl = configs["ssl"],
        )

    @staticmethod
    def _connect_to_db(host, port, database, username, password, path, ssl):
        """Initailize the DB client.

        Parameters
        ----------
        host : str
            InfluxDB host address
        port : int
            InfluxDB port number
        database : str
            InfluxDB database. Will be created if it does not exist already.
        username : str
            Username for the InfluxDB.
        password : str
            Password for the InfluxDB.
        path : str
            Path of InfluxDB on the server to connect to
        ssl : bool
            Use https to connect

        Returns
        ----------
        influxdb.client.InfluxDBClient
            InfluxDB client connected to specified database

        """
        if ssl:
            verify_ssl=True
        else:
            verify_ssl=False

        client = InfluxDBClient(host=host, port=port, username=username,
                                password=password, path=path, ssl=ssl,
                                verify_ssl=verify_ssl)
        db_list = client.get_list_database()
        db_names = [x['name'] for x in db_list]
        if database not in db_names:
            print(f"{database} DB doesn't exist, creating DB")
            client.create_database(database)
        client.switch_database(database)

        return client

    def check(self, field, observation, tags, log="obs_process_log"):
        """Check if monitored measurement has been reacorded already.

        All recorded measurement fields within the Monitor are tracked in a log
        within InfluxDB. This check will search this log with a search like::

            SELECT {field} FROM "log" WHERE observation = {observation} AND
                {tag1} = '{value1}' AND {tag2} = '{value2}';

        Parameters
        ----------
        field : str
            Measurement field to check calculation for, i.e. "white_noise_level"
        observation : str
            Observation ID
        tags : dict
            Other tags to included in AND search
        log : str
            Measurement name for the log within influxdb

        Returns
        -------
        bool
            True if calculation already performed, False otherwise

        """
        query_where = f"select {field} from \"{log}\" WHERE observation = '{observation}'"

        for tag_name, tag_value in tags.items():
            and_term = f" AND {tag_name} = '{tag_value}'"
            query_where += and_term

        result = self.client.query(query_where)

        if list(result.get_points(measurement=log)):
            print(f"field {field} for observation {observation} " +
                  f"and tags {tags} already recorded in {log}")
            return True

        return False

    @staticmethod
    def _build_single_line_entry(field, value, timestamp, tags, measurement):
        """Build a single line formatted string for insertion to InfluxDB.

        Creates a string of the form:
            '{measurement},{tag}={tag_value} {field}={value} {timestamp}'

        For many tags and tag values.

        Parameters
        ----------
        field : str
            Measurement field, i.e. "white_noise_level"
        value : float or int
            Value for the field
        timestamp : float
            Timestamp for the field value (can be None, which uses time of insertion to DB)
        tags : list of dict
            List of dictionaries containing tags for the InfluxDB
        measurement : str
            InfluxDB measurement to record to

        Returns
        -------
        str
            Single InfluxDB line formatted string.

        """

        # Single value/timestamp/tags
        influxdata = f'{measurement}'

        for tag, tag_value in tags.items():
            tag_string = f',{tag}={tag_value}'
            influxdata += tag_string

        influxdata += f' {field}={value}'

        if timestamp is not None:
            time_ns = int(timestamp*1e9)
            influxdata += f' {time_ns}'

        return influxdata

    def record(self, field, values, timestamps, tags, measurement, log='obs_process_log', log_tags=None):
        """Record a monitored statistic to the InfluxDB. Values not written to
        DB until ``Monitor.write()`` is called.

        Parameters
        ----------
        field : str
            Measurement field, i.e. "white_noise_level"
        values : list or np.array
            Values for the field for each unique set of tags and timestamps
        timestamps : list or np.array
            Timestamps for the field values
        tags : list of dict
            List of dictionaries containing tags for the InfluxDB
        measurement : str
            InfluxDB measurement to record to
        log : str
            InfluxDB measurement to use for logging completed calculation
        log_tags : list of dict
            Tags to use for the log, typically you won't want to record you've
            completed a calculation for each individual detector, but maybe some higher
            level group. If this is None tags will be used.

        """
        assert len(timestamps) == len(values) == len(tags)

        # Multi values/timestamps/tags
        for (value, ts, tag_dict) in zip(values, timestamps, tags):
            data_line = Monitor._build_single_line_entry(field, value, ts, tag_dict, measurement)
            self.queue.append(data_line)

        # Log into obs_process_log measurement in InfluxDB
        if log_tags is None:
            log_tags = tags

        log_msg = Monitor._build_single_line_entry(field, 1, None, log_tags, log)
        self.queue.append(log_msg)

    def write(self):
        """Write points to InfluxDB, clearing the queue."""
        self.client.write_points(self.queue, protocol='line')
        self.queue = []
