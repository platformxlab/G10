import random
import matplotlib.pyplot as plt

time = []
in_bws = []
out_bws = []
last_timestamp = 0
period = 50000

with open("../ResNet152-32/statistics/stat.txt.raw") as f:
  line = f.readline()
  while line:
    if line.startswith("@"):
      line_splitted = line.split()
      timestamp: int = int(line_splitted[1])
      delta_t: int = timestamp - last_timestamp
      last_timestamp = timestamp
      if (timestamp == 0):
        line = f.readline()
        continue
      time.append(timestamp / 1.2e9)

      in_pages = int(line_splitted[3])
      out_pages = int(line_splitted[5])
      in_bw = in_pages * 4096.0 / (float(delta_t) / 1.2e9) / 1e9
      out_bw = out_pages * 4096.0 / (float(delta_t) / 1.2e9) / 1e9
      in_bws.append(in_bw)
      out_bws.append(out_bw)
    line = f.readline()

# sel_start = random.randrange(0, len(time) - period)
print(len(time))
sel_start = 6249287
sel_end = sel_start + period

plt.plot(time[sel_start:sel_end], in_bws[sel_start:sel_end], "b-", label="In")
# plt.plot(time[sel_start:sel_end], out_bws[sel_start:sel_end], "r-", label="Out")
plt.legend()
plt.ylabel("PCIe BW (GBps)")
plt.xlabel("Execution time (s)")
plt.savefig("out.svg")

# with open("../ResNet152-32/statistics/stat.txt.raw") as f:
#   line = f.readline()
#   while line:
#     line_splitted = line.split()
#     if len(line_splitted) != 0 and line_splitted[0].endswith("slow_down"):
#       print(line)
#     line = f.readline()
