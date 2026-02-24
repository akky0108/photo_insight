import csv


class CSVPlugin:
    @staticmethod
    def read(file_path):
        with open(file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return list(reader)

    @staticmethod
    def write(file_path, data, header=None):
        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=header or data[0].keys())
            writer.writeheader()
            writer.writerows(data)
