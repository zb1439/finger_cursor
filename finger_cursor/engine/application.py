from finger_cursor.utils import Registry


APPLICATION = Registry("APPLICATION")


class Application:
    def __init__(self, cfg):
        self.cfg = cfg

    def terminate(self):
        """
        Will be called when the main process catches an ExitException
        """
        raise NotImplementedError

    def loop(self):
        """
        Runs the main loop of the application.
        Throws ExitException if necessary.
        """
        raise NotImplementedError

    def run(self):
        while True:
            self.loop()
            yield

    def async_run(self):
        raise NotImplementedError

    def async_check(self):
        return True


@APPLICATION.register()
class CursorControl(Application):  # default application
    def terminate(self):
        pass

    def loop(self):
        pass

    def async_run(self):
        pass
