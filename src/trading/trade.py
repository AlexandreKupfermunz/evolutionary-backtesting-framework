class Trade: 
    
    def __init__(self, entry_index, exit_index, direction, entry_price, exit_price, exit_reason, entry_timestamp, exit_timestamp, result):
        self.entry_index = entry_index
        self.exit_index = exit_index
        self.direction = direction
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.entry_timestamp = entry_timestamp
        self.exit_timestamp = exit_timestamp
        self.result = result
        self.window = None
        self.generation = None
        self.repetition_id = None

    def to_dict(self):

        row = {}

        row.update ({
            "entry_index": self.entry_index,
            "exit_index": self.exit_index,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "entry_timestamp": self.entry_timestamp,
            "exit_timestamp": self.exit_timestamp,
            "result": self.result
        })
        row.update(self.window.to_dict())
        row.update({"generation": self.generation,
                    "repetition_id": self.repetition_id})

        return row

def tradePrinter(trades):
    for i in range(len(trades)):
        print(f"Trade #{i+1}, entry index: {trades[i].entry_index}, direction: {trades[i].direction}, reason: {trades[i].exit_reason}, result: {trades[i].result}")
                      