import pandas as pd

df = pd.read_csv("naive_bayes_data.csv")

print(df["docs"])
print(df["class"])

docs_list = list()
docs = dict()
classes = dict()

size = len(df["docs"])
size -= 1
total_testing_rows = size
# print(f"Last: {df['docs'][size]}")
testing_row = df.iloc[size]
# print(testing_row)

# size of docs in each row
total_in_each_row = len(df["docs"][0].split(" "))
print(f"s -> {total_in_each_row}")

for doc in df["docs"]:
    docs_list += doc.split(" ")
    size -= 1
    if size == 0:
        break

for clas in df["class"]:
    if clas == "x":
        break
    key = clas.lower()
    value = classes.get(key)
    if value == None:
        classes.update({key: {"total_rows": 1}})
    else:
        classes.update({key: {"total_rows": value["total_rows"] + 1}})

for l in docs_list:
    key = l.lower()
    value = docs.get(key)
    if value == None:
        docs.update({key: 1})
    else:
        docs.update({key: value + 1})

print(docs)
print(len(docs))

print(classes)
print(len(classes))

vocab = len(docs)
for key in classes:
    classes[key].update(
        {"prior": round(classes[key]["total_rows"] / total_testing_rows, 4)}
    )
    classes[key].update({"total_docs": classes[key]["total_rows"] * total_in_each_row})

print(classes)
