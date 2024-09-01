import json

class JSONPlugin:

    @staticmethod
    def read(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def write(file_path, data, header=None):
        with open(file_path, 'w') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
