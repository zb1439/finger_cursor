from finger_cursor.engine import APPLICATION, Application

@APPLICATION.register()
class NSShaft(Application):
    def async_run(self):
        import subprocess
        p = subprocess.Popen(["open", "index.html"])
        self.p = p

    def terminate(self):
        if hasattr(self, "p"):
            self.p.terminate()

    def loop(self):
        pass

    def async_check(self):
        return True
