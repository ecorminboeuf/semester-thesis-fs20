import pandas as pd
import matplotlib.pyplot as plt

def read_columns(filename):
    with open(filename) as f:
        for l in f:
            if l[0] != '#':
                continue
            else:
                return l.strip('#').split()
        else:
            raise LookupError

def get_lines(filename, colnum):
    with open(filename) as f:
        for l in f:
            if l[0] == '1' or l[0] == '0':
                yield l.split(maxsplit=colnum - 1)

filename = '../../Difference/cos_potential/fourier_cos_eps025_smaller/cospoteps025-1594025682.pidstat'
columns = read_columns(filename)
#exclude = 'CPU', 'UID',
df = pd.DataFrame.from_records(
    get_lines(filename, len(columns)), columns=columns#, exclude=exclude
)
numcols = df.columns.drop('Command')
df[numcols] = df[numcols].apply(pd.to_numeric, errors='ignore')
#df['RSS'] = df.RSS / 1024  # Make MiB
df['Time'] = pd.to_datetime(df['Time'], utc=True, format = '%H:%M:%S')
df['Time'] = pd.to_numeric(df['Time'])
#df = df.set_index('Time')

#account for the fact that every column is removed by one
df['%CPU'] = df['CPU']
df['VSZ']  = df['RSS'] /(1024**2) #make GiB
df['RSS'] = df['%MEM'] /(1024**2) #make GiB
df.info()

units = ['1', 'GB', 'GB']
fig, axes = plt.subplots(len(df.PID.unique()), 3, figsize=(12, 8))
x_range = [df.index.min(), df.index.max()-120]
for i, pid in enumerate(df.PID.unique()):
    plt.figure(dpi=1200)
    subdf = df[df.PID == pid]
    title = ', '.join([f'PID {pid}', str(subdf.index.max() - subdf.index.min())])
    for j, col in enumerate(('%CPU', 'RSS', 'VSZ')):
        ax = subdf.plot(
            y=col, title=title if j == 0 else None, sharex=True
       )
        ax.legend(loc='upper right')
        ax.set_xlim(x_range)
        if col == 'VSZ':
            y_range = [37, 37.3]
            ax.set_ylim(y_range)
        ax.set_xlabel('time [arbitrary units]')
        ax.set_ylabel('%s ' %col + '[%s]' %units[j])
        plt.savefig('cospot_%s.pdf' %col)

plt.tight_layout()
#plt.savefig('tight.pdf')
plt.show()

