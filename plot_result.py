import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

folder = 'HedgeRatio/'
aa = np.load(folder+'cum_KF_bollinger_grid_high.npy')
a = np.load(folder+'cum_KF_approx_bollinger_gradient_high.npy')
aaa = np.load(folder+'cum_KF_bollinger_no_grid_high.npy')
b = np.load(folder+'cum_KNet_unsupervised_bollinger.npy')
c = np.load(folder+'cum_KNet_approx_bollinger_end_to_end.npy')
d = np.load(folder+'cum_KF_learnable_bollinger_high.npy')
e = np.load(folder+'cum_KNet_learnable_bollinger_end_to_end.npy')
f = np.load(folder+'cum_KF_approx_bollinger_gradient_high_rival.npy')
g = np.load(folder+'cum_rival.npy')
h = np.load(folder+'cum_ret_rival.npy')
df=pd.read_csv(folder+'forex1421'+'.csv', index_col='time')
df = df.iloc[2000:]

# df=pd.read_csv(folder+'rival'+'.csv', index_col='Date')
# df = df.iloc[1259:]

fig, ax = plt.subplots()
index = pd.date_range(datetime.date.fromisoformat(df.index[1].split()[0]),datetime.date.fromisoformat(df.index[-1].split()[0])+datetime.timedelta(days=1))
# index = range(len(a))
ax.plot(index, a, 'r', label = 'Kalman Filter Baseline')
# ax.plot(index, d, 'r', label = 'Kalman Filter + learnable bollinger')
# ax.plot(index, aa, 'b', label = 'KF (grid search)')
# ax.plot(index, aaa, 'r', label = 'KF (empirical value)')
# ax.plot(index, a, 'r', label = 'Kalman Filter baseline')
ax.plot(index, b, 'y', label = 'KalmanNet (Unsupervised)')
ax.plot(index, c, 'b', label = 'KalmanNet (maximize PnL)')
ax.plot(index, e, 'g', label = 'KalmanBOT (maximize PnL)')
# ax.plot(df.index, f, label = 'KF(gird search)')
# ax.plot(df.index, g, label = 'KNet(end-to-end)')
# ax.plot(index, i, 'r', label = 'Kalman Filter Baseline return')
# ax.plot(index, h, 'g', label = 'KalmanBOT (maximize PnL) return')

ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
ax.set_xlabel('time')
ax.set_ylabel('cumulative PnL')
# ax.set_title('when $\nu$ = '+str(vdB)+' dB')
ax.legend(fontsize=11)
ax.grid(True)

# dataFileName = 'forex_pnl_knet_approx_bollinger_end_to_end'
# dataFileName = 'rival_return'
dataFileName = 'forex_compare_all'
fig.savefig(folder+dataFileName+'.eps',transparent=True) 
fig.savefig(folder+dataFileName+'.png',transparent=True)