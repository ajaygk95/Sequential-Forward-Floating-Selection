import time


def timeit(method):  # referred to https://stackoverflow.com/questions/889900/accurate-timing-of-functions-in-python
    """
    time the runtime of a method

    :param method: function to be timed
    :return: function object
    """

    def get_run_time(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)

        print('\nalgorithm runtime: %2.2f seconds' % (time.time() - ts))
        return result

    return get_run_time

