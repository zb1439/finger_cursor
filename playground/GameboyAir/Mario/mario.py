from finger_cursor.engine import APPLICATION, Application

@APPLICATION.register()
class Mario(Application):
    def async_run(self):
        import subprocess
        p = subprocess.Popen(["nes_py", "-r", "./Super Mario Bros (E).nes"])
        self.p = p

    def terminate(self):
        if hasattr(self, "p"):
            self.p.terminate()

    def async_check(self):
        return self.p.poll() is None
