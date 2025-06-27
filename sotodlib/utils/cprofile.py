import cProfile, pstats, io

def cprofile(name):
    """Decorator to call CProfile on a function.
    """
    # Reference: https://stackoverflow.com/questions/5375624/a-decorator-that-profiles-a-method-call-and-logs-the-profiling-result
    def cprofile_func(func):
        def wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)
            s = io.StringIO()
            sortby = 'cumulative' # time spent by function and called subfunctions
            ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
            ps.print_stats(20) # print 10 longest calls
            print(f"{name} {func.__name__}: {s.getvalue()}")
            return retval

        return wrapper
    return cprofile_func