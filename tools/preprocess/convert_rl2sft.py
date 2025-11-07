import json
paths= ["/data/dlf/code/Field-Fidelity/data/rlhf/formatted/rlhf_formatted.jsonl","/data/dlf/code/Field-Fidelity/data/idk/data_format/idk_train_formatted_1k.jsonl"]

for path in paths:
    out_f = open(path.replace(".jsonl", "_sft.jsonl"), "w")
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            messages = item["messages"]
            messages.append({"role":"assistant","content":item["solution"]})
            record = {"messages":messages,"images":item["images"]}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_f.close()