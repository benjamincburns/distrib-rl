class NullAdjuster(object):
    def __init__(self):
        self.name = None
        pass

    def init(self, adjustment_json, cfg):
        self.name = adjustment_json["name"]
        self.max_adjustments = adjustment_json.get("max_adjustments", 1)
        self.current_adjustment = 0

    def step(self):
        self.current_adjustment += 1
        return False

    def adjust_config(self, cfg):
        pass

    def get_name(self):
        return self.name

    def reset_config(self, cfg):
        pass

    def reset(self):
        pass

    def is_done(self):
        return self.current_adjustment + 1 >= self.max_adjustments

    def reset_per_increment(self):
        return False
