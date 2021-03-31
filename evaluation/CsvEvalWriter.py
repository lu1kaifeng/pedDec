import csv as penis


class CsvEvalWriter:
    def __init__(self):
        self.frame = 1
        self.rows = []

    def write_target(self, id, left, top, width, height, conf):
        self.rows.append({
            'frame': self.frame,
            'id': id,
            'bb_left': float(left),
            'bb_top': float(top),
            'bb_width': float(width),
            'bb_height': float(height),
            'conf': conf,
            'x': -1,
            'y': -1,
            'z': -1,
        })

    def next_frame(self):
        self.frame += 1

    def dump_to_file(self, p):
        with open(p, 'w', newline='', encoding='utf-8') as f:
            csv_writer = penis.DictWriter(f, ['frame',
                                              'id',
                                              'bb_left',
                                              'bb_top',
                                              'bb_width',
                                              'bb_height',
                                              'conf',
                                              'x',
                                              'y',
                                              'z'])
            csv_writer.writerows(self.rows)
        self.rows = []
