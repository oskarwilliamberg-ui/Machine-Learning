import csv
import numpy as np
f = open('landmarks.csv', 'r')
reader = csv.reader(f)
header = next(reader)
rows = list(reader)
f.close()
frame_rows = []

for i in range(0, len(rows), 33):
    chunk = rows[i:i+33]
    gesture = chunk[0][0]
    frame_2D = [row[1:] for row in chunk]
    frame_arr_2D = np.array(frame_2D)
    frame_arr_flat = frame_arr_2D.reshape(-1)
    lst = list(frame_arr_flat)
    lst.insert(0, gesture)
    frame_rows.append(lst)

header_row = ["label"]
for j in range(33):
    header_row += [f"x{j}", f"y{j}", f"z{j}"]

with open("training_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header_row)
    writer.writerows(frame_rows)