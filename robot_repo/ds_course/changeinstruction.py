import json

with open("/home/ubuntu-user/robot_repo/ds_course/train.json") as f:
    data = json.load(f)

for item in data:
    item['instruction'] = "Please judge whether this news is true or false and give resson."

with open("/home/ubuntu-user/robot_repo/ds_course/train.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)