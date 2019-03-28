"""
Script to make plots of Position, Velocity, and Accelertation vs. time:
"""

import matplotlib.pyplot as plt
from matplotlib.mlab import psd, detrend_mean
import numpy as np
from scipy import signal


START_LINE = 26
#fname = "kv0.9_speed1_multiplemoves1_3_15_19.csv"
#fname = "repeatability.csv"
#fname = "1_micron_adjust_no_tuning.csv"
#fname = "kv0.9_speed1_1_micron_adjust_3_19_19.csv"
fname = "encoder_noise/encoder_noise_5min_3_26_19.csv"
#fname = "encoder_noise/encoder_noise.csv"
#fname = "accel2_kvstan10_kvmove1_3_19_19.csv"
#fname = "correct_ref_velo/accel2_1_micron_adj_3_26_19.csv"
print(fname)

f = open(fname, 'r')
f.readline() # line 1
f.readline() # line 2

start_time_line = f.readline() # line 3
start_time_array = start_time_line.split(',')
end_time_line = f.readline() # line 4
end_time_array = end_time_line.split(',')

start_time_split = start_time_array[5].split(':')
end_time_split = end_time_array[5].split(':')

start_hr = float(start_time_split[0])
start_min = float(start_time_split[1])
start_sec = float(start_time_split[2])
end_hr = float(end_time_split[0])
end_min = float(end_time_split[1])
end_sec = float(end_time_split[2])
start = (start_hr*3600) + (start_min*60.0) + start_sec
end = (end_hr*3600) + (end_min*60.0) + end_sec

delta_t = end - start
print('Measurement time:', delta_t)
f.close()

act_pos = []
set_pos = []
act_velo = []
set_velo = []
act_acc = []
set_acc = []
abs_gantry_err = []
# NOTE: seem to have 5x as many points for pos, velo, acc as abs_gantry_err

with open(fname, 'r') as f:
    line = f.readline()
    cnt = 1
    while line:
        line = f.readline()
        cnt += 1
        if cnt >= START_LINE:
            line_array = line.split(',')
            try:
                # For some reason abs_gantry_err has
                # less points than the others
                act_pos.append(float(line_array[1]))
                set_pos.append(float(line_array[3]))
                act_velo.append(float(line_array[5]))
                set_velo.append(float(line_array[7]))
                act_acc.append(float(line_array[9]))
                set_acc.append(float(line_array[11]))
                abs_gantry_err.append(float(line_array[13]))
            except:
                pass

# Going to numpy arrays!
act_pos = np.asarray(act_pos)
set_pos = np.asarray(set_pos)
act_velo = np.asarray(act_velo)
set_velo = np.asarray(set_velo)
act_acc = np.asarray(act_acc)
set_acc = np.asarray(set_acc)
abs_gantry_err = np.asarray(abs_gantry_err)

print('Number of Accel Points:', len(act_acc))
print('Number of Gantry Points:', len(abs_gantry_err))
print('Ratio:', len(act_acc)/len(abs_gantry_err))

tvals = np.linspace(0, delta_t, len(act_pos))
tvals_gantry = np.linspace(0, delta_t, len(abs_gantry_err))

f1, ax1 = plt.subplots()
ax1.plot(tvals, act_pos, label='Actual Position')
ax1.plot(tvals, set_pos, label='Set Position')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (mm)')
ax1.legend(loc='upper left')
ax1.set_title("Actual Position and Set Position")
f1.show()

f2, ax2 = plt.subplots()
ax2.plot(tvals, act_velo, label='Actual Velocity')
ax2.plot(tvals, set_velo, label='Set Velocity')
#ax2.plot(act_velo, label='Actual Velocity')
#ax2.plot(set_velo, label='Set Velocity')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (mm/s)')
ax2.legend(loc='upper right')
ax2.set_title("Actual Velocity and Set Velocity")
f2.show()

f3, ax3 = plt.subplots()
#ax3.plot(tvals_gantry, abs_gantry_err)
ax3.plot(abs_gantry_err)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Absolute Gantry Error (nm)')
ax3.set_title("Absolute Gantry Error")
f3.show()

# Compute dynamic gantry error
# Find index when we accelerate
# Divide by 5 -> Corresponding index for gantry data

# Compute average gantry difference before and after a move
# Consider not moving if abs(act_velo) < 0.002
#move_indeces = []
#VELOCITY_CUTOFF = 0.005 # in mm/s
#for i in range(len(act_velo)):
#    if abs(act_velo[i]) > VELOCITY_CUTOFF:
#        move_indeces.append(i)

#print(move_indeces[0])
# TODO: Can add way to automatically calculate later, for now just use
# visual


## Calculate fourier transform of actual position
# Getting Sampling Frequency
step_size = tvals[1] - tvals[0]
sampling_freq = len(np.arange(0, 1, step_size))

npoints = len(act_pos)
print(npoints)

# Using psd
ny, nx = psd(act_pos, NFFT=npoints, Fs=sampling_freq, detrend=detrend_mean,
             pad_to=16*npoints)
f4, ax4 = plt.subplots()
ax4.plot(nx, ny)
ax4.set_xlim(left=0, right=400)
f4.show()

# using fft
sp = np.fft.fft(act_pos)
freq_fft = np.fft.fftfreq(npoints, d=step_size)
f5, ax5 = plt.subplots()
ax5.plot(freq_fft, sp.real, freq_fft, sp.imag)
f5.show()

# using periodogram
freq_periodogram, Pxx_den = signal.periodogram(act_pos, sampling_freq)
f6, ax6 = plt.subplots()
ax6.plot(freq_periodogram, Pxx_den)
f6.show()

# repeatability.csv Gantry analysis:

#GANTRY_INCREASE_IDX1 = 2250
#GANTRY_PEAK_IDX1 = 2355
#GANTRY_SETTLED_IDX1 = 3400
#GANTRY_INCREASE_IDX2 = 5080
#GANTRY_PEAK_IDX2 = 5111
#GANTRY_SETTLED_IDX2 = 6100
#GANTRY_INCREASE_IDX3 = 6780
#GANTRY_PEAK_IDX3 = 6887
#GANTRY_SETTLED_IDX3 = 8000
#GANTRY_INCREASE_IDX4 = 9340
#GANTRY_PEAK_IDX4 = 9363
#GANTRY_SETTLED_IDX4 = 10300
#GANTRY_INCREASE_IDX5 = 11435
#GANTRY_PEAK_IDX5 = 11540
#GANTRY_SETTLED_IDX5 = 12679
#GANTRY_INCREASE_IDX6 = 14747
#GANTRY_PEAK_IDX6 = 14771
#GANTRY_SETTLED_IDX6 = 15707
#
#gantry_before1 = abs_gantry_err[:GANTRY_INCREASE_IDX1]
#gantry_after1 = abs_gantry_err[GANTRY_SETTLED_IDX1:GANTRY_INCREASE_IDX2]
#gantry_after2 = abs_gantry_err[GANTRY_SETTLED_IDX2:GANTRY_INCREASE_IDX3]
#gantry_after3 = abs_gantry_err[GANTRY_SETTLED_IDX3:GANTRY_INCREASE_IDX4]
#gantry_after4 = abs_gantry_err[GANTRY_SETTLED_IDX4:GANTRY_INCREASE_IDX5]
#gantry_after5 = abs_gantry_err[GANTRY_SETTLED_IDX5:GANTRY_INCREASE_IDX6]
#gantry_after6 = abs_gantry_err[GANTRY_SETTLED_IDX6:]
#
#gantry_before1_avg = np.mean(gantry_before1)
#gantry_after1_avg = np.mean(gantry_after1)
#gantry_after2_avg = np.mean(gantry_after2)
#gantry_after3_avg = np.mean(gantry_after3)
#gantry_after4_avg = np.mean(gantry_after4)
#gantry_after5_avg = np.mean(gantry_after5)
#gantry_after6_avg = np.mean(gantry_after6)
#
#static_err1 = gantry_after1_avg - gantry_before1_avg
#static_err2 = gantry_after2_avg - gantry_after1_avg
#static_err3 = gantry_after3_avg - gantry_after2_avg
#static_err4 = gantry_after4_avg - gantry_after3_avg
#static_err5 = gantry_after5_avg - gantry_after4_avg
#static_err6 = gantry_after6_avg - gantry_after5_avg
#
#dynamic_err1 = abs_gantry_err[GANTRY_PEAK_IDX1] - gantry_before1_avg
#dynamic_err2 = abs_gantry_err[GANTRY_PEAK_IDX2] - gantry_after1_avg
#dynamic_err3 = abs_gantry_err[GANTRY_PEAK_IDX3] - gantry_after2_avg
#dynamic_err4 = abs_gantry_err[GANTRY_PEAK_IDX4] - gantry_after3_avg
#dynamic_err5 = abs_gantry_err[GANTRY_PEAK_IDX5] - gantry_after4_avg
#dynamic_err6 = abs_gantry_err[GANTRY_PEAK_IDX6] - gantry_after5_avg
#
#print('static gantry error limit: 200 nm')
#print('Static gantry error 1: %s nm' % static_err1)
#print('Static gantry error 2: %s nm' % static_err2)
#print('Static gantry error 3: %s nm' % static_err3)
#print('Static gantry error 4: %s nm' % static_err4)
#print('Static gantry error 5: %s nm' % static_err5)
#print('Static gantry error 6: %s nm' % static_err6)
#
#print('')
#print('dynamic gantry error limit: 50000 nm')
#print('Dynamic gantry error 1: %s nm' % dynamic_err1)
#print('Dynamic gantry error 2: %s nm' % dynamic_err2)
#print('Dynamic gantry error 3: %s nm' % dynamic_err3)
#print('Dynamic gantry error 4: %s nm' % dynamic_err4)
#print('Dynamic gantry error 5: %s nm' % dynamic_err5)
#print('Dynamic gantry error 6: %s nm' % dynamic_err6)


# kv0.9_speed1_multiplemoves1_3_15_19.csv Gantry analysis:
#GANTRY_INCREASE_IDX1 = 1160
#GANTRY_PEAK_IDX1 = 1247
#GANTRY_SETTLED_IDX1 = 2550
#GANTRY_INCREASE_IDX2 = 3620
#GANTRY_PEAK_IDX2 = 3730
#GANTRY_SETTLED_IDX2 = 4580
#GANTRY_INCREASE_IDX3 = 6228
#GANTRY_PEAK_IDX3 = 6312
#GANTRY_SETTLED_IDX3 = 7160
#GANTRY_INCREASE_IDX4 = 8559
#GANTRY_PEAK_IDX4 = 8662
#GANTRY_SETTLED_IDX4 = 9297
#
#gantry_before1 = abs_gantry_err[:GANTRY_INCREASE_IDX1]
#gantry_after1 = abs_gantry_err[GANTRY_SETTLED_IDX1:GANTRY_INCREASE_IDX2]
#gantry_after2 = abs_gantry_err[GANTRY_SETTLED_IDX2:GANTRY_INCREASE_IDX3]
#gantry_after3 = abs_gantry_err[GANTRY_SETTLED_IDX3:GANTRY_INCREASE_IDX4]
#gantry_after4 = abs_gantry_err[GANTRY_SETTLED_IDX4:]
#
#gantry_before1_avg = np.mean(gantry_before1)
#gantry_after1_avg = np.mean(gantry_after1)
#gantry_after2_avg = np.mean(gantry_after2)
#gantry_after3_avg = np.mean(gantry_after3)
#gantry_after4_avg = np.mean(gantry_after4)
#
#static_err1 = gantry_after1_avg - gantry_before1_avg
#static_err2 = gantry_after2_avg - gantry_after1_avg
#static_err3 = gantry_after3_avg - gantry_after2_avg
#static_err4 = gantry_after4_avg - gantry_after3_avg
#
#dynamic_err1 = abs_gantry_err[GANTRY_PEAK_IDX1] - gantry_before1_avg
#dynamic_err2 = abs_gantry_err[GANTRY_PEAK_IDX2] - gantry_after1_avg
#dynamic_err3 = abs_gantry_err[GANTRY_PEAK_IDX3] - gantry_after2_avg
#dynamic_err4 = abs_gantry_err[GANTRY_PEAK_IDX4] - gantry_after3_avg
#
#print('static gantry error limit: 200 nm')
#print('Static gantry error 1: %s nm' % static_err1)
#print('Static gantry error 2: %s nm' % static_err2)
#print('Static gantry error 3: %s nm' % static_err3)
#print('Static gantry error 4: %s nm' % static_err4)
#
#print('')
#print('dynamic gantry error limit: 50000 nm')
#print('Dynamic gantry error 1: %s nm' % dynamic_err1)
#print('Dynamic gantry error 2: %s nm' % dynamic_err2)
#print('Dynamic gantry error 3: %s nm' % dynamic_err3)
#print('Dynamic gantry error 4: %s nm' % dynamic_err4)

# accel2_kvstan10_kvmove1_3_19_19.csv analysis
#GANTRY_INCREASE_IDX1 = 352
#GANTRY_PEAK_IDX1 = 452
#GANTRY_SETTLED_IDX1 = 592
#GANTRY_INCREASE_IDX2 = 995
#GANTRY_PEAK_IDX2 = 1000
#GANTRY_SETTLED_IDX2 = 1340
#GANTRY_INCREASE_IDX3 = 1558
#GANTRY_PEAK_IDX3 = 1662
#GANTRY_SETTLED_IDX3 = 1792
#GANTRY_INCREASE_IDX4 = 2126
#GANTRY_PEAK_IDX4 = 2131
#GANTRY_SETTLED_IDX4 = 2454
#GANTRY_INCREASE_IDX5 = 2741
#GANTRY_PEAK_IDX5 = 2839
#GANTRY_SETTLED_IDX5 = 3122
#GANTRY_INCREASE_IDX6 = 3369
#GANTRY_PEAK_IDX6 = 3378
#GANTRY_SETTLED_IDX6 = 3706
#
#gantry_before1 = abs_gantry_err[:GANTRY_INCREASE_IDX1]
#gantry_after1 = abs_gantry_err[GANTRY_SETTLED_IDX1:GANTRY_INCREASE_IDX2]
#gantry_after2 = abs_gantry_err[GANTRY_SETTLED_IDX2:GANTRY_INCREASE_IDX3]
#gantry_after3 = abs_gantry_err[GANTRY_SETTLED_IDX3:GANTRY_INCREASE_IDX4]
#gantry_after4 = abs_gantry_err[GANTRY_SETTLED_IDX4:GANTRY_INCREASE_IDX5]
#gantry_after5 = abs_gantry_err[GANTRY_SETTLED_IDX5:GANTRY_INCREASE_IDX6]
#gantry_after6 = abs_gantry_err[GANTRY_SETTLED_IDX6:]
#
#gantry_before1_avg = np.mean(gantry_before1)
#gantry_after1_avg = np.mean(gantry_after1)
#gantry_after2_avg = np.mean(gantry_after2)
#gantry_after3_avg = np.mean(gantry_after3)
#gantry_after4_avg = np.mean(gantry_after4)
#gantry_after5_avg = np.mean(gantry_after5)
#gantry_after6_avg = np.mean(gantry_after6)
#
#static_err1 = gantry_after1_avg - gantry_before1_avg
#static_err2 = gantry_after2_avg - gantry_after1_avg
#static_err3 = gantry_after3_avg - gantry_after2_avg
#static_err4 = gantry_after4_avg - gantry_after3_avg
#static_err5 = gantry_after5_avg - gantry_after4_avg
#static_err6 = gantry_after6_avg - gantry_after5_avg
#
#dynamic_err1 = abs_gantry_err[GANTRY_PEAK_IDX1] - gantry_before1_avg
#dynamic_err2 = abs_gantry_err[GANTRY_PEAK_IDX2] - gantry_after1_avg
#dynamic_err3 = abs_gantry_err[GANTRY_PEAK_IDX3] - gantry_after2_avg
#dynamic_err4 = abs_gantry_err[GANTRY_PEAK_IDX4] - gantry_after3_avg
#dynamic_err5 = abs_gantry_err[GANTRY_PEAK_IDX5] - gantry_after4_avg
#dynamic_err6 = abs_gantry_err[GANTRY_PEAK_IDX6] - gantry_after5_avg
#
#print('static gantry error limit: 200 nm')
#print('Static gantry error 1: %s nm' % static_err1)
#print('Static gantry error 2: %s nm' % static_err2)
#print('Static gantry error 3: %s nm' % static_err3)
#print('Static gantry error 4: %s nm' % static_err4)
#print('Static gantry error 5: %s nm' % static_err5)
#print('Static gantry error 6: %s nm' % static_err6)
#
#print('')
#print('dynamic gantry error limit: 50000 nm')
#print('Dynamic gantry error 1: %s nm' % dynamic_err1)
#print('Dynamic gantry error 2: %s nm' % dynamic_err2)
#print('Dynamic gantry error 3: %s nm' % dynamic_err3)
#print('Dynamic gantry error 4: %s nm' % dynamic_err4)
#print('Dynamic gantry error 5: %s nm' % dynamic_err5)
#print('Dynamic gantry error 6: %s nm' % dynamic_err6)


input('Press <return> to close')
