class WalkForwardWindow:

    def __init__(self, window_id, train_start, train_end, test_start, test_end):
        self.window_id = window_id
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

    def to_dict(self):

        return({
            "window_id": self.window_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end
        })