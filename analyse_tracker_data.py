import matplotlib
matplotlib.use('TkAgg')

import ROOT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import sndisplay as sn
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import time
from pathlib import Path
from numba import njit

input_root_file = 'red_612_output.root'
file = ROOT.TFile(f'/Users/casimirfisch/Desktop/Uni/SHP/{input_root_file}', 'READ')
tree = file.Get('event_tree')

# covariance could not be estimated - event 1824 and/or 1830

'''
more work to do:

- better filtering of the residuals in the cells to get better and neater error estimate 

15142 events in total
252 cells = 14 rows * 18 layers
20720 'tracks' with no filter (max-min, positive_timestamps)
13161 tracks with a max=20 and min=8 filter
7559 bad tracks
188168 total hits registered across all events, of which 123861 have positive r5, r6 and r0. (~65.8%)
'''

osdir = '/Users/casimirfisch/Desktop/Uni/SHP/Plots/'
histdir = 'histograms/'
hist3ddir = 'histograms 3d/'
filedir = 'Files/'
vertical_dir = 'vertical trajectory plots/'

first_level_file = osdir+filedir+f'tracks_firstlevel_max20_min8_minpos6.txt' # to be filled
second_level_file_10 = osdir+filedir+f'tracks_secondlevel_cutoff_0.10.txt'
second_level_file = osdir+filedir+f'tracks_secondlevel_cutoff_0.05.txt' # to be filled
# good_tracks_file = osdir+'good_tracks_with_fiterror_cutoff_0.2.txt' # to be filled
old_second_level_file = osdir+filedir+f'tracks_secondlevel_cutoff_0.4.txt'
good_tracks_file = second_level_file
# good_tracks_file = osdir+filedir+'tracks_with_cutoff_0.05.txt'
really_good_tracks = osdir+filedir+'tracks_secondlevel_cutoff_0.02.txt'
bb_decay_file = osdir+filedir+f'events_bb_decays_margin0.1.txt' #osdir+filedir+f'events_bb_decays_margin0.1.txt'
bb_decay_tracks_file = osdir+filedir+f'tracks_bb_decays_margin0.05.txt'
mean_std_residuals_file = osdir+filedir+'mean_residuals.txt'
mean_tpts_file = osdir+filedir+'mean_tpts.txt'
mean_tpts_goodtracks_file = osdir+filedir+'mean_tpts_goodtracks.txt'

empty_cells_goodtracks_tpt = [441,424,442,443,464,1468,1451,1462,1517,1510,1448,1457,1466,1502,1511]
empty_cells_goodtracks_res = [441,424,442,443,464,1468,1451,1462,1517,1510,1448,1457,1466,1502,468,1434,1435]

cells_one_hitonly = [400,437,1499,1510] 
# corresponding to 5813 fr, 8938 it, 9791 it, 10705 fr

high_resid_cells = [464,450,1461,1511] 
# could remove them from the final master residual histogram - but then again, are we introducing a bias??

tdc2sec = 12.5e-9
cell_radius = 0.022 #order of magnitude
cell_diameter = 2*cell_radius
cell_err = cell_radius
tracker_height = 3.

def change_rows(cell_nums):

    # row1, row2, side = 54, 55, 1
    cells_row1 = [1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511]
    cells_row2 = [1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520]

    new_cell_nums = []

    for cell in cell_nums:
        if cell in cells_row1:
            ind = cells_row1.index(cell)
            new_cell_nums.append(cells_row2[ind])
        elif cell in cells_row2:
            ind = cells_row2.index(cell)
            new_cell_nums.append(cells_row1[ind])
        else:
            new_cell_nums.append(cell)
    
    return np.array(new_cell_nums)   #  new_cell_nums

def filtering():

    print('\n*** First level of filtering ***\n')
    first_level_filtering()
    print('\n*** Second level of filtering ***\n')
    second_level_filtering()
    # third level: remove the bad cells? find the bad cells using the residuals
    # find_bb_decays()

def first_level_filtering(max_hits=20, min_hits=8, min_positive=6):

    # perform first-level filtering to the events in the ntuple file
    # and save the filtered tracks to a file, writing a track as [event_number side]

    # another filter could be linked to the minimum number of layers spanned - 3?

    filename = osdir+filedir+f'tracks_firstlevel_max{max_hits}_min{min_hits}_minpos{min_positive}.txt'
    f = open(filename, 'w')

    counter = 0

    for event in tree:

        event_number = event.event_number

        cell_nums = np.array(event.tracker_cell_num)
        cell_nums = change_rows(cell_nums)
        layers, rows, sides = cell_id(cell_nums)

        france, italy = sides==1, sides==0

        hits_fr = len(sides[france])
        hits_it = len(sides[italy])

        bot_arr = np.array(event.tracker_timestamp_r5)*tdc2sec
        top_arr = np.array(event.tracker_timestamp_r6)*tdc2sec
        an_arr = np.array(event.tracker_time_anode)
        bot_time = bot_arr - an_arr
        top_time = top_arr - an_arr

        # france 

        if max_hits >= hits_fr >= min_hits:
            good_hits = 0
            for i in range(hits_fr):
                if an_arr[france][i] > 0 and top_time[france][i] > 0 and bot_time[france][i] > 0:
                    good_hits+=1
            
            if good_hits >= min_positive:
                print(event_number, 'france')
                f.write('{} {}\n'.format(event_number, 'france'))
                counter+=1

        # italy

        if max_hits >= hits_it >= min_hits:
            good_hits = 0
            for i in range(hits_it):
                if an_arr[italy][i] > 0 and top_time[italy][i] > 0 and bot_time[italy][i] > 0:
                    good_hits+=1
            
            if good_hits >= min_positive:
                print(event_number, 'italy')
                f.write('{} {}\n'.format(event_number, 'italy'))
                counter+=1
            
    f.close()
    print('\ntotal tracks: ', counter)

def second_level_filtering(resid_cutoff=0.05):

    filename = first_level_file
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    newfile = osdir+filedir+f'tracks_secondlevel_cutoff_{resid_cutoff}.txt'
    f = open(newfile, 'w')
    counter = 0

    # run through the good tracks and introduce a filter that gets rid of instances where the error on the fit
    # is above a certain cutoff value (0.2 by inspection of tracks).

    for i in range(len(e_nums)):

        event_number = e_nums[i]
        tree.GetEntry(event_number)

        l, r, s, top, bot = filter_a_track(event_number, sides[i])
        # fit_err = get_fit_error(l, top, bot)
        residuals = np.abs(get_residuals(l, top, bot))
        
        if np.any(residuals > resid_cutoff):
            continue
    
        print(event_number, sides[i])
        f.write('{} {}\n'.format(e_nums[i], sides[i]))
        counter+=1

    f.close()

    print('total tracks without cutoff:', len(e_nums))
    print('total tracks with cutoff   :', counter)

def third_level_filtering(min_cell_hits=10):

    bad_cells = [1434, 1435]
    
    filename = second_level_file
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    for i in range(len(e_nums)):

        event_number = e_nums[i]
        tree.GetEntry(event_number)

        l, r, s, top, bot = filter_a_track(event_number, sides[i]) # maybe change the filter_a_track function to remove bad cells?

def find_bb_decays(margin=0.1):

    filename = second_level_file
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    # starting point within 10cm on either side of foil

    bb_decay_tracks_file = osdir+filedir+f'tracks_bb_decays_margin{margin}.txt'
    f = open(bb_decay_tracks_file, 'w')
    prev_event_number = 0
    counter = 0

    for i in range(len(e_nums)):

        event_number = e_nums[i]

        if event_number == prev_event_number: # two good tracks with same event number

            # # if there are events
            # if np.sum(layers[france]==0) >= 1 and np.sum(layers[italy]==0) >= 1:

            #     # if the starting row of the two tracks are the same, count it as a bb decay.
            #     same_row_arr = np.intersect1d(rows[france][layers[france]==0], rows[italy][layers[italy]==0])
            #     if same_row_arr.size != 0:

            l_fr, r_fr, s_fr, top_fr, bot_fr = filter_a_track(event_number, 'france')
            l_it, r_it, s_it, top_it, bot_it = filter_a_track(event_number, 'italy')
            vert_dist_fr = vertical_fractional(top_fr, bot_fr)
            horz_dist_fr = layers_to_distances(l_fr)
            vert_dist_it = vertical_fractional(top_it, bot_it)
            horz_dist_it = layers_to_distances(l_it)
            popt_fr, _ = curve_fit(linear, horz_dist_fr, vert_dist_fr)
            popt_it, _ = curve_fit(linear, horz_dist_it, vert_dist_it)

            track_atfoil_fr = popt_fr[1]
            track_atfoil_it = popt_it[1]

            print('france: {:.2f} m, italy {:.2f} m'.format(track_atfoil_fr, track_atfoil_it))
            
            # if the reconstructed tracks on either side of the source foil land with {margin} of each other
            if abs(track_atfoil_fr - track_atfoil_it) < margin:

                counter+=1
                # print(event_number, 'bb decay!')
                f.write(f'{event_number} france\n{event_number} italy\n'.format(event_number))

        prev_event_number = event_number

    f.close()
    print('\ntotal number of events:', counter)

def filter_events(max_hits=20, min_hits=8, param='one_row'):

    # could add a filter for the number of good cathode times

    if param == 'zero_filter':

        filename = osdir+f'tracks_{param}.txt'
        f = open(filename, 'w')

        counter = 0

        for event in tree:

            event_num = event.event_number

            cell_nums = np.array(event.tracker_cell_num)
            cell_nums = change_rows(cell_nums)
            layers, rows, sides = cell_id(cell_nums)

            france, italy = sides==1, sides==0

            hits_fr = len(sides[france])
            hits_it = len(sides[italy])

            if hits_fr > 0:
                counter += 1
                print(event_num, 'france')
                f.write('{} {}\n'.format(event_num, 'france'))
            if hits_it > 0:
                counter += 1
                print(event_num, 'italy')
                f.write('{} {}\n'.format(event_num, 'italy'))
        
        f.close()
        print(counter)

    elif param == 'bad_tracks':

        filename = osdir+f'{param}_(max{max_hits}_min{min_hits}).txt'
        f = open(filename, 'w')

        counter = 0

        for event in tree:

            event_num = event.event_number

            cell_nums = np.array(event.tracker_cell_num)
            cell_nums = change_rows(cell_nums)
            layers, rows, sides = cell_id(cell_nums)

            france, italy = sides==1, sides==0

            hits_fr = len(sides[france])
            hits_it = len(sides[italy])

            if max_hits < hits_fr or 0 < hits_fr < min_hits:
                counter += 1
                print(event_num, 'france')
                f.write('{} {}\n'.format(event_num, 'france'))
            if max_hits < hits_it or 0 < hits_it < min_hits:
                counter += 1
                print(event_num, 'italy')
                f.write('{} {}\n'.format(event_num, 'italy'))
        
        f.close()
        print(counter)

    elif param == 'no_timestamp_filter': # no cathode or anode filter

        filename = osdir+f'tracks_{param}_max{max_hits}_min{min_hits}.txt'
        f = open(filename, 'w')

        counter = 0

        for event in tree:

            event_num = event.event_number

            cell_nums = np.array(event.tracker_cell_num)
            cell_nums = change_rows(cell_nums)
            layers, rows, sides = cell_id(cell_nums)

            france, italy = sides==1, sides==0

            hits_fr = len(sides[france])
            hits_it = len(sides[italy])

            if max_hits >= hits_fr >= min_hits:
                counter += 1
                print(event_num, 'france')
                f.write('{} {}\n'.format(event_num, 'france'))
            if max_hits >= hits_it >= min_hits:
                counter += 1
                print(event_num, 'italy')
                f.write('{} {}\n'.format(event_num, 'italy'))
        
        f.close()
        print(counter)

    elif param == 'one_row':

        #good_events_fr, good_events_it = [], []

        filename = osdir+'good_events_one_row2.txt'
        f = open(filename, 'w')

        for event in tree:

            event_num = event.event_number

            cell_nums = np.array(event.tracker_cell_num)
            cell_nums = change_rows(cell_nums)
            layers, rows, sides = cell_id(cell_nums)

            r5_arr = np.array(event.tracker_timestamp_r5)
            r6_arr = np.array(event.tracker_timestamp_r6)
            tc_times = r6_arr*tdc2sec
            bc_times = r5_arr*tdc2sec
            an_times = np.array(event.tracker_time_anode)

            france, italy = sides==1, sides==0

            hits_fr = len(sides[france])
            hits_it = len(sides[italy])

            pos_times_fr = np.logical_and.reduce((tc_times[france]>0, bc_times[france]>0, an_times[france]>0))
            pos_times_it = np.logical_and.reduce((tc_times[italy]>0, bc_times[italy]>0, an_times[italy]>0))

            if max_hits >= hits_fr >= min_hits:
                # ensure that all hits are registered in the same row
                if np.all(pos_times_fr==True):
                    if np.all(rows[france] == rows[france][0]):
                        #good_events_fr.append(event_num)
                        print(event_num, 'france')
                        f.write('{} {}\n'.format(event_num, 'france'))
            if max_hits >= hits_it >= min_hits:
                # ensure that all hits are registered in the same row
                if np.all(pos_times_it==True):
                    if np.all(rows[italy] == rows[italy][0]):
                        #good_events_it.append(event_num)
                        print(event_num, 'italy')
                        f.write('{} {}\n'.format(event_num, 'italy'))
        
        f.close()

    elif param == 'many_rows':

        #good_events_fr, good_events_it = [], []

        filename = osdir+f'good_events_{param}.txt'
        f = open(filename, 'w')

        for event in tree:

            event_num = event.event_number

            cell_nums = np.array(event.tracker_cell_num)
            cell_nums = change_rows(cell_nums)
            layers, rows, sides = cell_id(cell_nums)

            r5_arr = np.array(event.tracker_timestamp_r5)
            r6_arr = np.array(event.tracker_timestamp_r6)
            an_times = np.array(event.tracker_time_anode)
            tc_times = r6_arr*tdc2sec
            bc_times = r5_arr*tdc2sec

            france, italy = sides==1, sides==0

            hits_fr = len(sides[france])
            hits_it = len(sides[italy])

            pos_times_fr = np.logical_and.reduce((tc_times[france]>0, bc_times[france]>0, an_times[france]>0))
            pos_times_it = np.logical_and.reduce((tc_times[italy]>0, bc_times[italy]>0, an_times[italy]>0))

            if max_hits >= hits_fr >= min_hits:
                # ensure that all hits are registered in the same row
                if np.all(pos_times_fr==True):
                    if np.any(rows[france] != rows[france][0]):
                        #good_events_fr.append(event_num)
                        print(event_num, 'france')
                        f.write('{} {}\n'.format(event_num, 'france'))
            if max_hits >= hits_it >= min_hits:
                # ensure that all hits are registered in the same row
                if np.all(pos_times_it==True):
                    if np.any(rows[italy] != rows[italy][0]):
                        #good_events_it.append(event_num)
                        print(event_num, 'italy')
                        f.write('{} {}\n'.format(event_num, 'italy'))

        f.close()

    elif param == 'good_all_filters':

        filename = osdir+f'tracks_{param}_max{max_hits}_min{min_hits}_updated2.txt'
        f = open(filename, 'w')

        counter = 0

        for event in tree:

            event_num = event.event_number

            cell_nums = np.array(event.tracker_cell_num)
            cell_nums = change_rows(cell_nums)
            layers, rows, sides = cell_id(cell_nums)

            france, italy = sides==1, sides==0

            hits_fr = len(sides[france])
            hits_it = len(sides[italy])

            min_positive = 5

            bot_arr = np.array(event.tracker_timestamp_r5)*tdc2sec
            top_arr = np.array(event.tracker_timestamp_r6)*tdc2sec
            an_arr = np.array(event.tracker_time_anode)
            bot_time = bot_arr - an_arr
            top_time = top_arr - an_arr 

            # if max_hits >= hits_fr >= min_hits:
            #     t1 = time.time()
            #     pos_times_fr = np.logical_and.reduce((bot_time[france]>0, top_time[france]>0, an_arr[france]>0))
            #     if np.sum(pos_times_fr) > min_positive:
            #         print(event_num, 'france')
            #         f.write('{} {}\n'.format(event_num, 'france'))
            #         counter+=1
            #     t2 = time.time()
            #     print('first method', t2-t1)
            # if max_hits >= hits_it >= min_hits:
            #     pos_times_it = np.logical_and.reduce((bot_time[italy]>0, top_time[italy]>0, an_arr[italy]>0))
            #     if np.sum(pos_times_it) > min_positive:
            #         print(event_num, 'italy')
            #         f.write('{} {}\n'.format(event_num, 'italy'))
            #         counter+=1

            # alternative with for loops --- about 2-3x faster!!

            # france 

            if max_hits >= hits_fr >= min_hits:
                # t1=time.time()
                good_hits = 0
                for i in range(hits_fr):
                    if an_arr[france][i] > 0 and top_time[france][i] > 0 and bot_time[france][i] > 0:
                        good_hits+=1
                
                if good_hits >= min_positive:
                    print(event_num, 'france')
                    f.write('{} {}\n'.format(event_num, 'france'))
                    counter+=1
                # t2=time.time()
                # print('second method', t2-t1)

            if max_hits >= hits_it >= min_hits:
                good_hits = 0
                for i in range(hits_it):
                    if an_arr[italy][i] > 0 and top_time[italy][i] > 0 and bot_time[italy][i] > 0:
                        good_hits+=1
                
                if good_hits >= min_positive:
                    print(event_num, 'italy')
                    f.write('{} {}\n'.format(event_num, 'italy'))
                    counter+=1
                
        f.close()
        print('\ntotal tracks: ', counter)

    elif param == 'bb_decays':

        # could loop through the good tracks file instead.

        margin = 0.1

        filename = osdir+f'{param}_events_same_z_margin{margin}.txt'
        f = open(filename, 'w')

        counter = 0

        for event in tree:

            event_num = event.event_number

            cell_nums = np.array(event.tracker_cell_num)
            cell_nums = change_rows(cell_nums)
            layers, rows, sides = cell_id(cell_nums)

            france, italy = sides==1, sides==0
            hits_fr = len(sides[france])
            hits_it = len(sides[italy])

            if np.sum(layers[france]==0) >= 1 and np.sum(layers[italy]==0) >= 1:

                # if the starting row of the two tracks are the same, count it as a bb decay.
                same_row_arr = np.intersect1d(rows[france][layers[france]==0], rows[italy][layers[italy]==0])
                if same_row_arr.size != 0:

                    if max_hits >= hits_fr >= min_hits and max_hits >= hits_it >= min_hits:

                        r5_arr = np.array(event.tracker_timestamp_r5)
                        r6_arr = np.array(event.tracker_timestamp_r6)
                        an_times = np.array(event.tracker_time_anode)
                        tc_times = r6_arr*tdc2sec
                        bc_times = r5_arr*tdc2sec

                        vert_dists = vertical_fractional(tc_times - an_times, bc_times - an_times)
                        # print(vert_dists)
                        same_row = same_row_arr[0]
                        # print(np.logical_and(rows==same_row, layers==0))

                        z_near_foil = vert_dists[np.logical_and(rows==same_row, layers==0)]

                        if abs(z_near_foil[0] - z_near_foil[1]) < margin:

                            counter+=1
                            print(event_num)
                            f.write('{}\n'.format(event_num))

        f.close()
        print('\ntotal number of events:', counter)

def vertical_fractional(top_dts, bot_dts):

    total_dts = top_dts + bot_dts
    return bot_dts/total_dts * tracker_height

def layers_to_distances(layers):
    # layer 0 is on point 1 (*cell_diameter)
    return (layers + .5) * cell_diameter

def tdt_errors(tdts, cells_list):

    # define the error as proportional to the distance of the tdt to the mean tdt for that cell
    filename = osdir+'mean_tdts.txt'
    data = np.loadtxt(filename, dtype=float)
    cell_nums, mean_tdts, tdt_stdevs = data[:,0].astype(int), data[:,1], data[:,2]

    errors = []

    for i in range(len(tdts)):

        # mean_tdt = mean_tdts[cell_nums==cells_list[i]][0]
        # # if mean_tdt == np.nan: delta_tdt = 0.01 else:
        # delta_tdt = abs(tdts[i] - mean_tdt)
        # # if delta_tdt == 0: delta_tdt =  None
        # errors.append(delta_tdt)

        tdt_err = tdt_stdevs[cell_nums==cells_list[i]][0]
        errors.append(tdt_err)

    return errors

def resid_errors(cells_list):

    filename = mean_std_residuals_file
    data = np.loadtxt(filename, dtype=float)
    cell_nums, mean_resids, stdev_resids = data[:,0].astype(int), data[:,1], data[:,2]

    errors = []

    for cell in cells_list:
        err = stdev_resids[cell_nums==cell][0] / 2 # mean_resids[cell_nums==cell][0]
        if err == 0: # only one residual value, stdev = 0
            # take the mean of the absolute values instead, so that the error bar is non-zero
            err = mean_resids[cell_nums==cell][0] 
        errors.append(err)
    
    return errors

def vertical_error(vert_distances, total_dts, cells_list, tpt_err=False):

    if tpt_err == True:

        tdt_errs = tdt_errors(total_dts, cells_list)
        return tdt_errs * vert_distances / total_dts 
    # 
    # error on the vertical fractional distances

    # vert_errs = []
    # for i in range(len(tdt_errs)):
    #     if tdt_errs[i]==None: vert_errs.append(None)
    #     else: vert_errs.append(tdt_errs[i] * vert_distances[i] / total_dts[i])
    # return vert_errs

    else:
        resid_errs = resid_errors(cells_list)
        return resid_errs

def vertical_distances(tc_time, bc_time):

    tracker_h = tracker_height
    total_drift_time = (tc_time + bc_time)
    fraction_top, fraction_bot = tc_time/total_drift_time, bc_time/total_drift_time
    t_distance, b_distance = fraction_top*tracker_h, fraction_bot*tracker_h

    return t_distance, b_distance

def linear(x, a, b):
    return a*x + b

def plot_vertical_fractional(vert_dist, horz_dist, vert_err, event_number, side, tpt_err=False, interpolate=False, invert=False):

    # t_dist and b_dist are mirror images of each other because they are calculated as fractions
    # only consider one of them (bottom, since the electron goes up when the bottom fraction increases)
    popt, pcov = curve_fit(linear, horz_dist, vert_dist, sigma=vert_err)
    perr = np.sqrt(np.diag(pcov))

    line_xs = np.arange(0,10)*cell_diameter
    cell_boundaries = np.arange(0.5,8.5)*cell_diameter + cell_radius

    fig, ax = plt.subplots(tight_layout=True, figsize=(5,6))

    # title = 'Vertical (fractional) trajectory against horizontal distance\nfor track [{} - {}]'.format(event_number, side)
    ax.errorbar(horz_dist, vert_dist, yerr=vert_err, xerr=cell_err, fmt='o', color='k') #355C7D
    ax.plot(line_xs, linear(line_xs, *popt), '-', color='r') #F67280
    ax.set_title('Event {}, {}'.format(event_number, side.capitalize()))
    # fig.suptitle(title)
    ax.set_xlabel('layer axis /m') #'distance from source foil /m', fontsize='large'
    ax.set_ylabel('z /m', rotation='horizontal') #fontsize='large'
    ax.yaxis.set_label_coords(-0.05,1.)
    ax.set_xlim(line_xs[0], line_xs[-1])
    if invert:
        ax.invert_xaxis()

    for boundary in cell_boundaries:
        ax.axvline(boundary, color='k', linestyle='dashed', alpha=0.5, linewidth=1)

    # ax.text(0.05, 0.05, 'error of fit: {:.4e}'.format(perr[0]), transform=ax.transAxes)

    if tpt_err==True:
        fig.savefig(osdir+vertical_dir+'track{0}_{1}_tpterr.png'.format(event_number, side))
    elif interpolate==True:
        fig.savefig(osdir+vertical_dir+'track{0}_{1}_interpolated.png'.format(event_number, side))
    else:
        fig.savefig(osdir+vertical_dir+'track{0}_{1}.png'.format(event_number, side))

    plt.show()

def filter_event(event_number):

    tree.GetEntry(event_number)

    cell_nums = np.array(tree.tracker_cell_num)
    cell_nums = change_rows(cell_nums)
    layers, rows, sides = cell_id(cell_nums)
    an_times = np.array(tree.tracker_time_anode)
    r5_arr = np.array(tree.tracker_timestamp_r5)
    r6_arr = np.array(tree.tracker_timestamp_r6)
    tc_times = r6_arr*tdc2sec
    bc_times = r5_arr*tdc2sec
    cath_filter = np.logical_and(tc_times>0, bc_times>0)
    positive_filter = np.logical_and(cath_filter, an_times>0)

    l, r, s = layers[positive_filter], rows[positive_filter], sides[positive_filter]
    top_dt, bot_dt = (tc_times - an_times)[positive_filter], (bc_times - an_times)[positive_filter]

    return l, r, s, top_dt*1e6, bot_dt*1e6

def filter_a_track(event_number, side):

    tree.GetEntry(event_number)

    cell_nums = np.array(tree.tracker_cell_num)
    cell_nums = change_rows(cell_nums)
    layers, rows, sides = cell_id(cell_nums)
    an_times = np.array(tree.tracker_time_anode)
    r5_arr = np.array(tree.tracker_timestamp_r5)
    r6_arr = np.array(tree.tracker_timestamp_r6)
    tc_times = r6_arr*tdc2sec - an_times
    bc_times = r5_arr*tdc2sec - an_times

    if side=='france' or  side==1:
        side_filter = sides==1
    else:
        side_filter = sides==0

    positive_filter = (tc_times[side_filter]>0) * (bc_times[side_filter]>0) * (an_times[side_filter]>0)
    l = layers.copy()
    l = l[side_filter][positive_filter]
    inds = l.argsort() # sort the different hits from the innermost to the outermost layer

    l_sorted = layers[side_filter][positive_filter][inds]
    r_sorted = rows[side_filter][positive_filter][inds]
    s_sorted = sides[side_filter][positive_filter][inds]
    tc_sorted = tc_times[side_filter][positive_filter][inds]
    bc_sorted = bc_times[side_filter][positive_filter][inds]
    # an_sorted = an_times[side_filter][positive_filter][inds]

    return l_sorted, r_sorted, s_sorted, tc_sorted*1e6, bc_sorted*1e6 # work in microseconds

def plot_tracks_3D(event_number_list, side):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface()
    ax.set_ylim(42, 56)
    ax.set_xlim(0, 8)
    ax.set_zlim(0, tracker_height)
    ax.set_xlabel('Layer number')
    ax.set_ylabel('Row number')
    ax.set_zlabel('Vertical distance')
    title = '3D tracks [side: {}]'.format(side)

    for event_num in event_number_list:

        l_sorted, r_sorted, s_sorted, tdt_sorted, bdt_sorted = filter_a_track(event_num, side)
        b_dist_avg, err = vertical_dist_err(tdt_sorted, bdt_sorted)
        ax.plot(l_sorted, r_sorted, b_dist_avg, label=f'event {event_num}')

    ax.legend()
    fig.suptitle(title)
    fig.savefig(osdir+'3D_events:{0}_{1}.png'.format(event_number_list, side))
    plt.show()

def plot_tracks_3D_2sides(e_list, sides_list, filename, scatter=False, with_legend=True):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ys = np.arange(42, 56.1, 0.1)
    zs = np.arange(0, tracker_height+0.1, 0.1)
    Y, Z = np.meshgrid(ys, zs)
    X = np.zeros_like(Y)
    ax.plot_surface(X, Y, Z, alpha=0.3)

    ax.set_ylim(42, 56)
    ax.set_xlim(-8.5, 8.5)
    ax.set_xticks(np.arange(-8.5,9.5))
    ax.set_yticks(np.arange(42,57))
    ax.set_zlim(0, tracker_height)
    ax.set_xlabel('Layer number')
    ax.set_ylabel('Row number')
    ax.set_zlabel('Vertical distance')
    # title = f'3D tracks [{filename}]'
    ax.view_init(60,-20)

    for i in range(len(e_list)):

        l_sorted, r_sorted, s_sorted, tdt_sorted, bdt_sorted = filter_a_track(e_list[i], sides_list[i])
        vert_dist = vertical_fractional(tdt_sorted, bdt_sorted)
        l_sorted = l_sorted + .5 # so that 0 is not doubly  occupied

        if sides_list[i] == 'italy':
            l_sorted = -l_sorted # convert to other side
            track_label = f'event {e_list[i]} (it)'
        else:
            track_label = f'event {e_list[i]} (fr)'

        if scatter == True:
            ax.scatter(l_sorted, r_sorted, vert_dist, label=track_label)
        else:
            ax.plot(l_sorted, r_sorted, vert_dist, label=track_label)

    if with_legend==True:
        ax.legend()

    # fig.suptitle(title)
    fig.savefig(osdir+'3D_events_{}.png'.format(filename))
    plt.show()

def plot_3D_2sides_events(e_nums_list:list, scatter=True):

    fig = plt.figure(figsize=(10,8)) #
    ax = fig.add_subplot(projection='3d')

    xs = np.arange(41.5, 56.5, 0.1)
    zs = np.arange(0, tracker_height+0.1, 0.1)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.3, color='k', linewidth=10)

    major_xticks = np.arange(42,56)
    minor_xticks = np.arange(41.5,56.5)
    major_yticks = np.arange(-8.5,9.5)
    minor_yticks = np.arange(-9,10)
    layer_ticks = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    ax.set_xlim(41.5,55.5)
    ax.set_ylim(-8.5,8.5)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)
    # ax.set_xticks(minor_xticks, minor = True)
    # ax.set_yticks(minor_yticks, minor = True)
    ax.set_yticklabels(layer_ticks)
    ax.xaxis.grid(which='minor')
    ax.yaxis.grid(which='minor')

    ax.text(48.5, -4, 0.1*tracker_height, 'Italy',  fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')
    ax.text(48.5, +4, 0.1*tracker_height, 'France', fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')

    ax.set_zlim(0, tracker_height)
    ax.set_xlabel('Row number')
    ax.set_ylabel('Layer number')
    ax.set_zlabel('Vertical distance')
    # title = f'3D plot of event {event_number}'

    for event_number in e_nums_list:
        l_sorted, r_sorted, s_sorted, tdt_sorted, bdt_sorted = filter_event(event_number)
        vert_dist = vertical_fractional(tdt_sorted, bdt_sorted)
        l_sorted = adjusted_layers(l_sorted, s_sorted)

        if scatter == True:
            ax.scatter(r_sorted, l_sorted, vert_dist, label=f'{event_number}')
        else:
            ax.plot(r_sorted, l_sorted, vert_dist)

    # ax.legend()
    ax.view_init(25,-15)
    # ax.set_title(title)
    fig.savefig(osdir+'3D_2sides_events{}.png'.format(e_nums_list), transparent=True)
    plt.show()

def plot_3D_2sides_event(event_number, scatter=True):

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(projection='3d')

    xs = np.arange(41.5, 56.5, 0.1)
    zs = np.arange(0, tracker_height+0.1, 0.1)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.3, color='k', linewidth=10)

    major_xticks = np.arange(42,56)
    minor_xticks = np.arange(41.5,56.5)
    major_yticks = np.arange(-8.5,9.5)
    minor_yticks = np.arange(-9,10)
    layer_ticks = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    ax.set_xlim(41.5,55.5)
    ax.set_ylim(-8.5,8.5)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)
    # ax.set_xticks(minor_xticks, minor = True)
    # ax.set_yticks(minor_yticks, minor = True)
    ax.set_yticklabels(layer_ticks)
    ax.xaxis.grid(which='minor')
    ax.yaxis.grid(which='minor')

    ax.text(48.5, -4, 0.1*tracker_height, 'Italy',  fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')
    ax.text(48.5, +4, 0.1*tracker_height, 'France', fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')

    ax.set_zlim(0, tracker_height)
    ax.set_xlabel('Row number')
    ax.set_ylabel('Layer number')
    ax.set_zlabel('Vertical distance')
    # title = f'3D plot of event {event_number}'

    l_sorted, r_sorted, s_sorted, tdt_sorted, bdt_sorted = filter_event(event_number)
    vert_dist = vertical_fractional(tdt_sorted, bdt_sorted)
    l_sorted = adjusted_layers(l_sorted, s_sorted)

    if scatter == True:
        ax.scatter(r_sorted, l_sorted, vert_dist, color='r')
    else:
        ax.plot(r_sorted, l_sorted, vert_dist)

    ax.view_init(25,-15)
    # ax.set_title(title)
    fig.savefig(osdir+'3D_2sides_event{}.png'.format(event_number))
    plt.show()

def scan_track_nofilter(event_number, side):

    tree.GetEntry(event_number)

    cell_nums = np.array(tree.tracker_cell_num)
    cell_nums = change_rows(cell_nums)
    layers, rows, sides = cell_id(cell_nums)
    an_times = np.array(tree.tracker_time_anode)
    r5_arr = np.array(tree.tracker_timestamp_r5)
    r6_arr = np.array(tree.tracker_timestamp_r6)
    tc_times = r6_arr*tdc2sec
    bc_times = r5_arr*tdc2sec

    if side=='france':
        side_filter = sides==1
    else:
        side_filter = sides==0

    l = layers.copy()[side_filter]
    inds = l.argsort() # sort the different hits from the innermost to the outermost layer

    l_sorted = layers[side_filter][inds]
    r_sorted = rows[side_filter][inds]
    s_sorted = sides[side_filter][inds]
    # r5_sorted = r5_arr[side_filter][inds]
    # r6_sorted = r6_arr[side_filter][inds]
    tc_sorted = tc_times[side_filter][inds]
    bc_sorted = bc_times[side_filter][inds]
    an_sorted = an_times[side_filter][inds]
    tdt_sorted = (tc_sorted - an_sorted)*1e6 # microseconds
    bdt_sorted = (bc_sorted - an_sorted)*1e6 # microseconds
    total_drift_time = tdt_sorted + bdt_sorted

    print('\n*** Event {0} ({1}) *** (no filter)\n'.format(event_number, side))
    for i in range(len(l_sorted)):
        print('''layer {0} / row {1} / side {2} / an_time {3:.3e} / top_c {4:.3e} / bot_c {5:.3e} / top_dt {6:.3f} / bot_dt {7:.3f} / total {8:.3f}
'''.format(l_sorted[i], r_sorted[i], s_sorted[i], an_sorted[i], tc_sorted[i], bc_sorted[i], tdt_sorted[i], bdt_sorted[i], total_drift_time[i]))

def scan_track_longer(event_number, side='france', with_cath_filter=True):

    tree.GetEntry(event_number)

    cell_nums = np.array(tree.tracker_cell_num)
    cell_nums = change_rows(cell_nums)
    layers, rows, sides = cell_id(cell_nums)
    an_times = np.array(tree.tracker_time_anode)
    r5_arr = np.array(tree.tracker_timestamp_r5)
    r6_arr = np.array(tree.tracker_timestamp_r6)
    tc_times = r6_arr*tdc2sec
    bc_times = r5_arr*tdc2sec

    if side=='france':
        side_filter = sides==1
    else:
        side_filter = sides==0

    if with_cath_filter==True:

        cath_filter = np.logical_and(tc_times[side_filter]>0, bc_times[side_filter]>0)
        l = layers.copy()
        l = l[side_filter][cath_filter]
        inds = l.argsort() # sort the different hits from the innermost to the outermost layer

        l_sorted = layers[side_filter][cath_filter][inds]
        r_sorted = rows[side_filter][cath_filter][inds]
        s_sorted = sides[side_filter][cath_filter][inds]
        tc_sorted = tc_times[side_filter][cath_filter][inds]
        bc_sorted = bc_times[side_filter][cath_filter][inds]
        an_sorted = an_times[side_filter][cath_filter][inds]
        tdt_sorted = tc_sorted - an_sorted
        bdt_sorted = bc_sorted - an_sorted
        total_drift_time = tdt_sorted + bdt_sorted
        t_distance, b_distance = vertical_distances(tdt_sorted, bdt_sorted)

        print('\n*** Event {0} ({1}) ***\n'.format(event_number, side))
        for i in range(len(l_sorted)):
            print('''layer {0} / row {1} / side {2} / an_time {3:.8f} / tdt {4:.3e} / bdt {5:.3e} / total {6:.4e} / t_distance {7:.2f} / b_distance {8:.2f}
'''.format(l_sorted[i], r_sorted[i], s_sorted[i], an_sorted[i], tdt_sorted[i], bdt_sorted[i], total_drift_time[i], t_distance[i], b_distance[i]))

    else: # no cathode filtering

        l = layers.copy()
        l = l[side_filter]
        inds = l.argsort()
        l_sorted = layers[side_filter][inds]
        r_sorted = rows[side_filter][inds]
        s_sorted = sides[side_filter][inds]
        tc_sorted = tc_times[side_filter][inds]
        bc_sorted = bc_times[side_filter][inds]
        an_sorted = an_times[side_filter][inds]
        r6_sorted = r6_arr[side_filter][inds]
        r5_sorted = r5_arr[side_filter][inds]
        tdt_sorted = tc_sorted - an_sorted
        bdt_sorted = bc_sorted - an_sorted
        t_distance, b_distance = vertical_distances(tdt_sorted, bdt_sorted)

        print('\n*** Event {0} ({1}) ***\n'.format(event_number, side))
        for i in range(len(l_sorted)):
            print('''layer {0} / row {1} / side {2} / an_time {3:.8f} / r6 {4:.6e} / r5 {5:.6e} / t_distance {6:.4f} / b_distance {7:.4f}
'''.format(l_sorted[i], r_sorted[i], s_sorted[i], an_sorted[i], r6_sorted[i], r5_sorted[i], t_distance[i], b_distance[i]))

def scan_and_plot(event_number, side='france', tpt_err=False, interpolate=False, invert=False):

    if is_a_good_track(event_number, side): print('\ngood track!')
    else: print('\nbad track!')

    l_sorted, r_sorted, s_sorted, tdt_sorted, bdt_sorted = filter_a_track(event_number, side)
    cells_sorted = cell_num_(l_sorted, r_sorted, s_sorted)
    total_dts = tdt_sorted + bdt_sorted
    vert_dist = vertical_fractional(tdt_sorted, bdt_sorted)
    horz_dist = layers_to_distances(l_sorted)
    vert_errs = vertical_error(vert_dist, total_dts, cells_sorted, tpt_err=tpt_err)

    print('\n*** Event {0} ({1}) ***\n'.format(event_number, side))
    for i in range(len(l_sorted)):
        print('''cell {} / layer {} / row {} / side {} / top {:.3f} / bot {:.3f} / total {:.4f} / vertical {:.3f} +/- {:.3f}
'''.format(cells_sorted[i], l_sorted[i], r_sorted[i], s_sorted[i], tdt_sorted[i], bdt_sorted[i], total_dts[i], vert_dist[i], vert_errs[i]))

    plot_vertical_fractional(vert_dist, horz_dist, vert_errs, event_number, side, tpt_err=tpt_err, interpolate=interpolate, invert=invert)

def PlotEvent(event_number=1253, with_input=False):

    # myStyle = ROOT.TStyle('MyStyle','My graphics style')
    # myStyle.SetCanvasColor(ROOT.kWaterMelon)
    # ROOT.gROOT.SetStyle('MyStyle')
    if with_input==True:
        event_number = int(input('Event Number: '))

    ROOT.gStyle.SetOptStat(False)
    ROOT.gStyle.SetPalette(ROOT.kWaterMelon)

    tree.GetEntry(event_number)

    cell_nums = np.array(tree.tracker_cell_num)
    cell_nums = change_rows(cell_nums)
    layers, rows, sides = cell_id(cell_nums)
    hits = len(layers)

    title = 'Event {} ({} hits)'.format(event_number, hits)
    trackerMap = ROOT.TH2C("tracker_map", title, 14, 42, 56, 18, -9, 9)

    for i in range(hits):

        if sides[i]==1:
            adjusted_layer = layers[i]+0.5
        else:
            adjusted_layer = -layers[i]-0.5
        
        trackerMap.Fill(rows[i], adjusted_layer)

    canv = ROOT.TCanvas("canv",title,600,600)
    canv.SetGrid()
    trackerMap.Draw("COL")

    sourceFoil = ROOT.TLine(42,0,56,0)
    sourceFoil.SetLineColor(ROOT.kBlack)
    sourceFoil.SetLineWidth(4)
    sourceFoil.Draw("SAME")

    italyText = ROOT.TPaveText(47,-5,51,-3)
    italyText.AddText("Italy")
    italyText.SetFillStyle(0)
    italyText.SetBorderSize(0)
    italyText.Draw()
    
    franceText = ROOT.TPaveText(47,3,51,5)
    franceText.AddText("France")
    franceText.SetFillStyle(0)
    franceText.SetBorderSize(0)
    franceText.Draw()

    plotFileName = osdir+'event_{}_tracks.pdf'.format(event_number)

    canv.Print(plotFileName)

def plot_xy_event(event_number):

    n_rows, n_layers = 14, 18
    grid = np.zeros((n_rows, n_layers))

    tree.GetEntry(event_number)

    cell_nums = np.array(tree.tracker_cell_num)
    cell_nums = change_rows(cell_nums)
    layers, rows, sides = cell_id(cell_nums)
    layers = adjusted_layers(layers, sides)
    an_times = np.array(tree.tracker_time_anode)
    bot_times = np.array(tree.tracker_timestamp_r5)*tdc2sec - an_times
    top_times = np.array(tree.tracker_timestamp_r6)*tdc2sec - an_times

    positive_filter = (an_times>0) * (top_times>0) * (bot_times>0)
    negative_filter = np.invert(positive_filter)

    fig, ax = plt.subplots(figsize=(5,6), tight_layout=True)
    ax.set_aspect(1)

    major_xticks = np.arange(42,56)
    minor_xticks = np.arange(41.5,56.5)
    major_yticks = np.arange(-8.5,9.5)
    minor_yticks = np.arange(-9,10)
    layer_ticks = [8,7,6,5,4,3,2,1,0,0,1,2,3,4,5,6,7,8]

    good_label = 'good hits'
    bad_label = 'hits filtered out'
    ax.scatter(rows[positive_filter], layers[positive_filter], s=100, marker='o', color='g', alpha=0.5, label=good_label)
    ax.scatter(rows[negative_filter], layers[negative_filter], s=100, marker='o', color='r', alpha=0.5, label=bad_label)
    ax.plot(np.arange(41,57), np.zeros_like(np.arange(41,57)), 'k', linewidth=2) # source foil line
    ax.set_xlim(41.5,55.5)
    ax.set_ylim(-9,9)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)
    ax.set_xticks(minor_xticks, minor = True)
    ax.set_yticks(minor_yticks, minor = True)
    ax.set_yticklabels(layer_ticks)
    ax.set_ylabel('layer number')
    ax.set_xlabel('row number')
    ax.grid(which='minor')
    ax.set_title(f'Event {event_number}')
    # ax.text(0.5, 0.75, 'France', transform=ax.transAxes, fontsize='xx-large', horizontalalignment='center', verticalalignment='center', fontfamily='serif', fontweight='bold')
    # ax.text(0.5, 0.25, 'Italy', transform=ax.transAxes, fontsize='xx-large', horizontalalignment='center', verticalalignment='center', fontfamily='serif', fontweight='bold')
    ax.text(-.08, 0.75, 'France', transform=ax.transAxes, fontsize='x-large', horizontalalignment='center', verticalalignment='center', fontfamily='serif', fontweight='bold', rotation=90)
    ax.text(-.08, 0.25, 'Italy', transform=ax.transAxes, fontsize='x-large', horizontalalignment='center', verticalalignment='center', fontfamily='serif', fontweight='bold', rotation=90)
    
    # ax.legend()

    fig.savefig(osdir+f'xyplane_event{event_number}.png')

    plt.show()

def cell_id(cellnum):
    cell_side = cellnum // (9 * 113)
    cell_row = cellnum % (9 * 113) // 9
    cell_layer = cellnum % (9 * 113) % 9
    return cell_layer, cell_row, cell_side

def cell_num_(layer, row, side):
    return side*113*9 + row*9 + layer

def sn_PlotEvent(event_number):

    tree.GetEntry(event_number)

    cell_nums = np.array(tree.tracker_cell_num)
    cell_nums = change_rows(cell_nums)
    layers, rows, sides = cell_id(cell_nums)
    hits = len(layers)

    sntracker = sn.tracker(f'track_{event_number}', with_palette=True)
    sntracker.draw_cellid_label()
    sntracker.draw_content_label('{}')

    for i_cell in cell_nums:
        sntracker.fill(i_cell)

    sntracker.setrange(0, 35)

    sntracker.draw()
    sntracker.save(osdir)

def tracks_from_file(filename):

    data = np.loadtxt(filename, dtype=str)
    events_nums, sides = data[:,0].astype(int), data[:,1]

    return events_nums, sides

def tdt_cell(events_arr, sides_arr, cell_num, file_abbrev):

    tdts = []

    for i in range(len(events_arr)):

        l_sorted, r_sorted, s_sorted, tdt_sorted, bdt_sorted = filter_a_track(events_arr[i], sides_arr[i])
        cell_num_sorted = cell_num_(l_sorted, r_sorted, s_sorted)
        if cell_num not in cell_num_sorted:
            continue
        total_drift_times = tdt_sorted + bdt_sorted
        tdt = total_drift_times[cell_num_sorted==cell_num] # select the total drift times that are in the cell.
        
        if tdt.size == True: # ensure array is not empty
            
            if tdt[0] < 1e3: # if one of the cathode values are negative, the tdt will be very large (>1e3)

                tdts.append(tdt[0])
            
                if tdt[0] > 60: # the odd ones, usually
                    print(events_arr[i], tdt[0])

    tdts = np.array(tdts)

    mean_tdt = np.mean(tdts)
    print('\nMean = {:.2f}\n'.format(mean_tdt))

    fig = plt.figure()
    plt.hist(tdts, bins=30, color='darksalmon', histtype='step')
    plt.axvline(mean_tdt, color='k', linestyle='dashed', linewidth=1)
    plt.xlabel('total drift time /µs')
    plt.ylabel('counts')
    plt.title('Total drift times for cell {0} {1} [{2}]'.format(cell_num, cell_id(cell_num), file_abbrev))
    fig.savefig(osdir+f'hist_tdt_cell_{cell_num}.png')
    plt.show()

def tdt_cell_all_events(cell_num, plot_or_not=False, blip=0, plot_odd_events=False, low_limit=50, high_limit=80):

    l, r, s = cell_id(cell_num)

    count_neghits = 0

    tdts, events = [], []
    if plot_odd_events: 
        odd_events, odd_tdts = [], []

    for event in tree:

        # one event will only have one hit in a given cell, if any at all
        event_number = event.event_number
        cell_nums = np.array(event.tracker_cell_num)
        cell_nums = change_rows(cell_nums)
        if cell_num not in cell_nums:
            continue
        if len(cell_nums) > 50: # filter out certain events that have almost all of the cells triggered - 'noisy events'
            continue

        anode_time = np.array(event.tracker_time_anode)[cell_nums==cell_num][0]
        bot_cath_time = np.array(event.tracker_timestamp_r5)[cell_nums==cell_num][0]*tdc2sec - anode_time
        top_cath_time = np.array(event.tracker_timestamp_r6)[cell_nums==cell_num][0]*tdc2sec - anode_time

        # require all three to be positive numbers
        if anode_time < 0 or bot_cath_time < 0 or top_cath_time < 0:
            count_neghits+=1
            # print(event_number, 'negative times - out!!')
            continue

        total_drift_time = (top_cath_time+bot_cath_time)*1e6

        if  blip != 0: # to look for outliers
            print('Event {} / tdt: {:.3f}'.format(event_number, total_drift_time))
            if total_drift_time < blip:
                print('blip!')
                if plot_odd_events: 
                    odd_events.append(event_number)
                    odd_tdts.append(total_drift_time)

        if high_limit > total_drift_time > low_limit:
            tdts.append(total_drift_time)
            events.append(event_number)

    print('\nnegative hits:', count_neghits)

    mean_tdt, stdev = np.mean(tdts), np.std(tdts)
    print('\nCell Number: {}\nMean = {:.2f} / stdev = {:.2f}\n'.format(cell_num, mean_tdt, stdev))

    if plot_or_not==True:

        fig = plt.figure()
        plt.hist(tdts, bins=30, color='darksalmon', histtype='step')
        plt.axvline(mean_tdt, color='k', linestyle='dashed', linewidth=1)
        plt.xlabel('propagation time /µs')
        plt.ylabel('counts')
        plt.title('Total propagation times for cell {0} ({1}.{2}.{3})\nN = {4} | mean = {5:.2f}'.format(cell_num, l, r, s, len(tdts), mean_tdt))
        fig.savefig(osdir+f'hist_tdt_cell_{cell_num}_all_events.png')
        plt.show()
    
    if plot_odd_events:

        fig, ax = plt.subplots()
        # ax.plot(odd_events, odd_tdts)
        ax.hist(odd_events, bins=80, histtype='step', color='teal')
        ax.set_xlabel('event number')
        ax.set_ylabel('counts')
        ax.set_title('Events with a suprisingly low propagation time\nCell {0} ({1}.{2}.{3})'.format(cell_num, l, r, s))
        fig.savefig(osdir+f'hist_oddtdts_cell_{cell_num}.png')
        plt.show()
    
    else: return mean_tdt

def tdt_cell_good_tracks(cell_num):

    layer, row, side = cell_id(cell_num)
    side_str = 'france' if side == 1 else 'italy'

    filename = good_tracks_file
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    tdts = []

    for i in range(len(e_nums)):

        event_number = e_nums[i]
        tree.GetEntry(event_number)

        if sides[i] != side_str: continue

        l, r, s, top, bot = filter_a_track(event_number, sides[i])
        cell_nums = cell_num_(l, r, s)
        if cell_num not in cell_nums: continue
        print(event_number, sides[i])
        tdt = (top + bot)[cell_nums==cell_num][0]
        tdts.append(tdt)
    
    mean_tdt, stdev = np.mean(tdts), np.std(tdts)
    print('\nCell Number: {}\nMean = {:.2f} / stdev = {:.2f}\n'.format(cell_num, mean_tdt, stdev))

    fig = plt.figure()
    plt.hist(tdts, bins=30, color='darksalmon', histtype='step')
    plt.axvline(mean_tdt, color='k', linestyle='dashed', linewidth=1)
    plt.xlabel('propagation time /µs')
    plt.ylabel('counts')
    plt.title('Total propagation times for cell {0} ({1}.{2}.{3})\nN = {4} | mean = {5:.2f} [good tracks]'.format(cell_num, layer, row, side, len(tdts), mean_tdt))
    fig.savefig(osdir+f'hist_tdt_cell_{cell_num}_good_tracks.png')
    plt.show()

def hist_tpt_fromfile_cell(cell_num, param='all events', min_val=0, max_val=0):

    layer, row, side = cell_id(cell_num)
    side_str = 'France' if side==1 else 'Italy'

    if param == 'all events':
        tpt_file = osdir+f'TPT/tpt_allevents_cell_{cell_num}.txt'
        tpts = np.loadtxt(tpt_file)
    elif param == 'good tracks':
        tpt_file = osdir+f'TPT/tpt_goodtracks_cell_{cell_num}.txt'
        tpts = np.loadtxt(tpt_file)
    
    if min_val != 0 and max_val != 0:
        good_indices = (tpts > min_val) * (tpts < max_val)
        tpts = tpts[good_indices]

    mean_tpt, tpt_std = np.mean(tpts), np.std(tpts)
    print('\nCell Number: {}\nMean propagation time = {:.4f} / stdev = {:.4f}\n'.format(cell_num, mean_tpt, tpt_std))

    fig = plt.figure(figsize=(5,6), tight_layout=True)
    plt.hist(tpts, bins=20, color='darkcyan', histtype='stepfilled', edgecolor='k')
    # plt.axvline(mean_tpt, color='k', linestyle='dashed', linewidth=1)
    plt.xlabel('propagation time /µs')
    plt.ylabel('counts')

    if param == 'all events':
        plt.title('Cell {} (row {}, layer {}, {})'.format(cell_num, row, layer, side_str))
        fig.savefig(osdir+histdir+f'hist_tpt_cell_{cell_num}_all_events.png')
    elif param == 'good tracks':
        plt.title('Total propagation times for cell {0} ({1}.{2}.{3})\nN = {4} | mean = {5:.2f} [good tracks]'.format(cell_num, layer, row, side, len(tpts), mean_tpt))
        fig.savefig(osdir+histdir+f'hist_tpt_cell_{cell_num}_good_tracks.png')

    plt.show()

def get_all_cell_nums():

    rows = np.arange(42,56)
    layers = np.arange(0,9)
    sides = np.array([0,1])

    all_cell_nums = []

    for side in sides:
        for layer in layers:
            for row in rows:
                all_cell_nums.append(cell_num_(layer, row, side))

    # print(all_cell_nums)
    return all_cell_nums

def write_mean_tpt_to_file(param='all events'):

    if param == 'all events':

        all_cell_nums = get_all_cell_nums()
        tdt_mega_list = [[] for i in range(len(all_cell_nums))]
        # mean_tdts = np.zeros(len(all_cell_nums))

        for event in tree:

            print(event.event_number)
            cell_nums = list(event.tracker_cell_num)
            cell_nums = change_rows(cell_nums)

            if len(cell_nums) > 50: 
            # filter out certain events that have almost all of the cells triggered
                continue

            r5, r6, an = list(event.tracker_timestamp_r5), list(event.tracker_timestamp_r6), list(event.tracker_time_anode)
            # bot_arr = np.array(event.tracker_timestamp_r5)*tdc2sec
            # top_arr = np.array(event.tracker_timestamp_r6)*tdc2sec
            # an_arr = np.array(event.tracker_time_anode)
            # bot_time = bot_arr - an_arr
            # top_time = top_arr - an_arr 

            for i in range(len(cell_nums)):

                if an[i] < 0 or (r5[i]*tdc2sec - an[i]) < 0 or (r6[i]*tdc2sec - an[i]) < 0:
                    continue

                tdt = ((r5[i]+r6[i])*tdc2sec - 2*an[i])*1e6
                ind = all_cell_nums.index(cell_nums[i])
                tdt_mega_list[ind].append(tdt)

        f = open(mean_tpts_file, 'w')

        for j in range(len(all_cell_nums)):

            if len(tdt_mega_list[j]) != 0:
                mean_tdt = np.mean(tdt_mega_list[j])
                tdt_stdev = np.std(tdt_mega_list[j])
                # mean_tdts[j] = mean_tdt
                f.write('{} {:.4f} {:.4f}\n'.format(all_cell_nums[j], mean_tdt, tdt_stdev))

                tdt_file = osdir+f'TPT/tpt_allevents_cell_{all_cell_nums[j]}.txt'
                np.savetxt(tdt_file, tdt_mega_list[j])

            else:
                f.write('{} nan nan\n'.format(all_cell_nums[j]))
                print(all_cell_nums[j], 'empty!\n')

        f.close()

    # if param == 'all events':

    #     all_cell_nums = get_all_cell_nums()
    #     tdt_mega_list = [[] for i in range(len(all_cell_nums))]
    #     # mean_tdts = np.zeros(len(all_cell_nums))

    #     for event in tree:

    #         print(event.event_number)
    #         cell_nums = np.array(event.tracker_cell_num)
    #         cell_nums = change_rows(cell_nums)

    #         if 8 > len(cell_nums) or len(cell_nums) > 40: 
    #         # filter out certain events that have almost all of the cells triggered
    #             continue

    #         bot_arr = np.array(event.tracker_timestamp_r5)*tdc2sec
    #         top_arr = np.array(event.tracker_timestamp_r6)*tdc2sec
    #         an_arr = np.array(event.tracker_time_anode)
    #         bot_time = bot_arr - an_arr
    #         top_time = top_arr - an_arr 

    #         for i in range(len(cell_nums)):

    #             if an_arr[i] < 0 or bot_time[i] < 0 or top_time[i] < 0:
    #                 continue

    #             tdt = (top_time[i]+bot_time[i])*1e6
    #             ind = all_cell_nums.index(cell_nums[i])
    #             tdt_mega_list[ind].append(tdt)

    #     f = open(mean_tpts_file, 'w')

    #     for j in range(len(all_cell_nums)):

    #         if len(tdt_mega_list[j]) != 0:
    #             mean_tdt = np.mean(tdt_mega_list[j])
    #             tdt_stdev = np.std(tdt_mega_list[j])
    #             # mean_tdts[j] = mean_tdt
    #             f.write('{} {:.4f} {:.4f}\n'.format(all_cell_nums[j], mean_tdt, tdt_stdev))

    #             tdt_file = osdir+f'TPT/tpt_allevents_cell_{all_cell_nums[j]}.txt'
    #             np.savetxt(tdt_file, tdt_mega_list[j])

    #         else:
    #             f.write('{} nan nan\n'.format(all_cell_nums[j]))
    #             print(all_cell_nums[j], 'empty!\n')

    #     f.close()
    
    elif param == 'good tracks':

        filename = good_tracks_file
        data = np.loadtxt(filename, dtype=str)
        e_nums, sides = data[:,0].astype(int), data[:,1]

        all_cell_nums = get_all_cell_nums()
        tdt_mega_list = [[] for i in range(len(all_cell_nums))]
        mean_tdts = np.zeros(len(all_cell_nums))

        for i in range(len(e_nums)):

            event_number = e_nums[i]
            tree.GetEntry(event_number)
            print(event_number, sides[i])
            l, r, s, top, bot = filter_a_track(event_number, sides[i])
            cell_nums = cell_num_(l, r, s)

            if 1517 in cell_nums: print('\nThis one!\n')

            tdts = top + bot
            for j in range(len(cell_nums)):

                ind = all_cell_nums.index(cell_nums[j])
                tdt_mega_list[ind].append(tdts[j])

        f = open(mean_tpts_goodtracks_file, 'w')

        for z in range(len(all_cell_nums)):

            if len(tdt_mega_list[z]) != 0:
                mean_tdt = np.mean(tdt_mega_list[z])
                tdt_stdev = np.std(tdt_mega_list[z])
                mean_tdts[z] = mean_tdt
                f.write('{} {:.4f} {:.4f}\n'.format(all_cell_nums[z], mean_tdt, tdt_stdev))

                tdt_file = osdir+f'TPT/tpt_goodtracks_cell_{all_cell_nums[z]}.txt'
                np.savetxt(tdt_file, tdt_mega_list[z])

            else:
                f.write('{} nan nan\n'.format(all_cell_nums[z]))
                print(all_cell_nums[z], 'empty!\n')

        f.close()

def colour_plot_tdts():

    # more stuff to do here if you've got the time

    data = np.loadtxt(osdir+'mean_tdts.txt', dtype=float)
    cell_nums, mean_tdts, stdevs = np.array(data[:,0].astype(int)), np.array(data[:,1]), np.array(data[:,2])
    num_bars = len(cell_nums)
    l, r, s = cell_id(cell_nums)
    l = adjusted_layers(l, s)

    bad_indices = np.isnan(mean_tdts) 
    good_indices = ~bad_indices  # remove the nan values

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    zmin = 40
    zmax = mean_tdts[good_indices].max()

    xs = np.arange(41.5, 55.5, 0.1)
    zs = np.arange(zmin, int(zmax)+0.1, 0.1)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.5, color='k')

    major_xticks = np.arange(42,56)
    minor_xticks = np.arange(41.5,56.5)
    major_yticks = np.arange(-8.5,9.5)
    minor_yticks = np.arange(-9,10)
    layer_ticks = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    ax.set_xlim(41.5,55.5)
    ax.set_ylim(-8.5,8.5)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)
    # ax.set_xticks(minor_xticks, minor = True)
    # ax.set_yticks(minor_yticks, minor = True)
    ax.set_yticklabels(layer_ticks)
    ax.xaxis.grid(which='minor')
    ax.yaxis.grid(which='minor')

    # ax.set_zlim(0, tracker_height)
    ax.set_xlabel('Row number')
    ax.set_ylabel('Layer number')
    ax.set_zlabel('Mean total propagation time')
    title = 'Histogram of total propagation times across all cells'

    x, y, z = r[good_indices]-.25, l[good_indices]-.25, zmin #np.zeros(num_bars)
    dx, dy, dz = np.ones_like(z)*1/2, np.ones_like(z)*1/2, mean_tdts[good_indices] - zmin

    # print(x,y,z,dx,dy,dz)

    cmap = cm.get_cmap('viridis')
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    rgba = [cmap((k-min_height)/max_height) for k in dz]

    ax.bar3d(x, y, z, dx, dy, dz, zsort='average', color=rgba) 
    ax.text(42, -4, 80, 'Italy',  fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')
    ax.text(42, +4, 80, 'France', fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')

    ax.view_init(60,-20)

    fig.savefig(osdir+hist3ddir+'hist3D_mean_tdts.png')

    plt.show()

def hist2D_tpts(param='all'):

    all_cell_nums = np.array(get_all_cell_nums())
    l, r, s = cell_id(all_cell_nums)
    l = adjusted_layers(l, s)
    mean_tpts = []

    for cell_num in all_cell_nums:

        layer, row, side = cell_id(cell_num)

        if param == 'all':
            tpt_file = osdir+f'TPT/tpt_allevents_cell_{cell_num}.txt'
        elif param == 'good tracks':
            tpt_file = osdir+f'TPT/tpt_goodtracks_cell_{cell_num}.txt'
        if Path(tpt_file).exists() == False: 
            mean_tpts.append(np.nan)
            print(f'bad cell: {cell_num} ({layer}.{row}.{side})')
            continue
        tpts = np.loadtxt(tpt_file)

        # filtering for the cell if necessary

        # if cell_num == 1434 or cell_num == 1435: 
        #     mean_resids.append(np.nan)
        # elif residuals.size > min_resids: 
        #     mean_resids.append(np.mean(residuals))
        # else: 
        mean_tpts.append(np.mean(tpts))
        # print('Cell {} - count too low: {}'.format(cell_num, residuals.size))
    
    mean_tpts = np.array(mean_tpts)
    bad_indices = np.isnan(mean_tpts) 
    good_indices = ~bad_indices  # remove the nan values
    # overall_mean = np.mean(mean_tpts[good_indices])

    # print('\nMean residual error for a cell: {:.4f}\n'.format(overall_mean))

    layer_ticks = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    rows = np.arange(42,56)
    layers = np.arange(-8.5,9.5)
    Z_ = mean_tpts.reshape(18, 14)
    Z = np.empty_like(Z_)
    Z[:9,:] = np.flipud(Z_[:9,:])
    Z[9:,:] = Z_[9:,:]

    fig, ax = plt.subplots(figsize=(6,6), tight_layout=True)
    title = f'Mean cathode total propagation time for each cell - \'{param}\''

    c = ax.pcolormesh(rows, layers, Z, edgecolors='k', linewidths=1, cmap='plasma')
    ax.plot(np.arange(41.5,56.5), np.zeros_like(np.arange(41.5,56.5)), 'k', linewidth=4) # source foil line
    cbar = fig.colorbar(c, ax=ax, ticks=[55,60,65,70,75,80])
    cbar.ax.set_yticklabels(['55µs','60µs','65µs','70µs','75µs','80µs'])
    # ax.set_title(title, fontsize='large')
    ax.set_xlabel('rows')
    ax.set_ylabel('layers')
    ax.set_yticks(layers, layer_ticks)
    ax.set_xticks(rows, rows)
    ax.text(-.08, 0.75, 'France', transform=ax.transAxes, fontsize='x-large', horizontalalignment='center', verticalalignment='center', fontfamily='serif', rotation=90)
    ax.text(-.08, 0.25, 'Italy', transform=ax.transAxes, fontsize='x-large', horizontalalignment='center', verticalalignment='center', fontfamily='serif', rotation=90)
    # ax.set_aspect(1)

    fig.savefig(osdir+hist3ddir+f'hist2d_tpts_{param}.png')

    plt.show()

def hist3D_tdts():

    data = np.loadtxt(osdir+'mean_tdts.txt', dtype=float)
    cell_nums, mean_tdts, stdevs = np.array(data[:,0].astype(int)), np.array(data[:,1]), np.array(data[:,2])
    num_bars = len(cell_nums)
    l, r, s = cell_id(cell_nums)
    l = adjusted_layers(l, s)

    bad_indices = np.isnan(mean_tdts) 
    good_indices = ~bad_indices  # remove the nan values

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    zmin = 40
    zmax = mean_tdts[good_indices].max()

    xs = np.arange(41.5, 55.5, 0.1)
    zs = np.arange(zmin, int(zmax)+0.1, 0.1)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.5, color='k')

    major_xticks = np.arange(42,56)
    minor_xticks = np.arange(41.5,56.5)
    major_yticks = np.arange(-8.5,9.5)
    minor_yticks = np.arange(-9,10)
    layer_ticks = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    ax.set_xlim(41.5,55.5)
    ax.set_ylim(-8.5,8.5)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)
    # ax.set_xticks(minor_xticks, minor = True)
    # ax.set_yticks(minor_yticks, minor = True)
    ax.set_yticklabels(layer_ticks)
    ax.xaxis.grid(which='minor')
    ax.yaxis.grid(which='minor')

    # ax.set_zlim(0, tracker_height)
    ax.set_xlabel('Row number')
    ax.set_ylabel('Layer number')
    ax.set_zlabel('Mean total propagation time')
    title = 'Histogram of total propagation times across all cells'

    x, y, z = r[good_indices]-.25, l[good_indices]-.25, zmin #np.zeros(num_bars)
    dx, dy, dz = np.ones_like(z)*1/2, np.ones_like(z)*1/2, mean_tdts[good_indices] - zmin

    # print(x,y,z,dx,dy,dz)

    cmap = cm.get_cmap('viridis')
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    rgba = [cmap((k-min_height)/max_height) for k in dz]

    ax.bar3d(x, y, z, dx, dy, dz, zsort='average', color=rgba) 
    ax.text(42, -4, 80, 'Italy',  fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')
    ax.text(42, +4, 80, 'France', fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')

    ax.view_init(60,-20)

    fig.savefig(osdir+hist3ddir+'hist3D_mean_tdts.png')

    plt.show()

def hits_all_cells_to_file(param='all'):

    # function for number of hits total for all cells  
    # all valid cells (with both cathode times)? or all cells generally

    filename = osdir+f'lrs_{param}_hits_forhist_withlimit.txt'

    f=open(filename, 'w')
     
    for event in tree:

        print(event.event_number)
        cell_nums = np.array(event.tracker_cell_num)
        cell_nums = change_rows(cell_nums)
        if len(cell_nums) > 50: # preventive fitlering to ignore events that feature too many hits
            continue
        layers, rows, sides = cell_id(cell_nums)

        if param == 'all': # no filtering, fast analysis

            for i in range(len(layers)): 

                f.write(f'{layers[i]} {rows[i]} {sides[i]}\n')

        elif param == 'positive': # filtering the positive anode and cathode times only, slower analysis

            anode_times = np.array(event.tracker_time_anode)
            r5_arr = np.array(event.tracker_timestamp_r5)
            r6_arr = np.array(event.tracker_timestamp_r6)

            for i in range(len(layers)):

                if anode_times[i] < 0 or r5_arr[i] < 0 or r6_arr[i] < 0: # discard any negative ones
                    continue

                f.write(f'{layers[i]} {rows[i]} {sides[i]}\n')

        elif param == 'negative': # param = negative

            anode_times = np.array(event.tracker_time_anode)
            r5_arr = np.array(event.tracker_timestamp_r5)
            r6_arr = np.array(event.tracker_timestamp_r6)

            for i in range(len(layers)):

                if anode_times[i] > 0 and r5_arr[i] > 0 and r6_arr[i] > 0: # discard the positive ones
                    continue

                f.write(f'{layers[i]} {rows[i]} {sides[i]}\n')

    f.close()

def write_hits_cells_tofile(param='all', with_limit=False):

    filename = osdir+filedir+f'hits_cells_{param}.txt'
    if with_limit: filename = osdir+filedir+f'hits_cells_{param}_max50.txt'
    f = open(filename, 'w')

    all_cell_nums = np.array(get_all_cell_nums())
    hits_counts = np.zeros_like(all_cell_nums)

    if param == 'all':

        for event in tree:
            cell_nums = np.array(tree.tracker_cell_num)
            if with_limit: 
                if len(cell_nums) > 50: continue
            cell_nums = change_rows(cell_nums)
            print(event.event_number)
            print()
            print(cell_nums)
            for cell in cell_nums:
                hits_counts[all_cell_nums==cell] += 1

        for i in range(len(all_cell_nums)):
            f.write('{} {}\n'.format(all_cell_nums[i], hits_counts[i]))

    elif param == 'positive':

        for event in tree:
            cell_nums = np.array(event.tracker_cell_num)
            if with_limit: 
                if len(cell_nums) > 50: continue
            cell_nums = change_rows(cell_nums)
            bot_arr = np.array(event.tracker_timestamp_r5)*tdc2sec
            top_arr = np.array(event.tracker_timestamp_r6)*tdc2sec
            an_arr = np.array(event.tracker_time_anode)
            bot_time = bot_arr - an_arr
            top_time = top_arr - an_arr 
            print(event.event_number)

            for i in range(len(cell_nums)):
                if bot_time[i] > 0 and top_time[i] > 0 and an_arr[i] > 0:
                    # f.write(f'{cell_nums[i]}\n')
                    hits_counts[all_cell_nums==cell_nums[i]] += 1

        for j in range(len(all_cell_nums)):
            f.write('{} {}\n'.format(all_cell_nums[j], hits_counts[j]))

    elif param == 'positive anodes':

        for event in tree:
            cell_nums = list(event.tracker_cell_num) #np.array(event.tracker_cell_num)
            if with_limit: 
                if len(cell_nums) > 50: continue
            cell_nums = change_rows(cell_nums)
            
            an_arr = list(event.tracker_time_anode)
            print(event.event_number)

            for i in range(len(cell_nums)):
                if an_arr[i] > 0:
                    # f.write(f'{cell_nums[i]}\n')
                    hits_counts[all_cell_nums==cell_nums[i]] += 1

        print(hits_counts)
        for j in range(len(all_cell_nums)):
            f.write('{} {}\n'.format(all_cell_nums[j], hits_counts[j]))

    elif param  == 'goodtracks':

        goodtracks = good_tracks_file
        data = np.loadtxt(goodtracks, dtype=str)
        e_nums, sides = data[:,0].astype(int), data[:,1]

        for i in range(len(e_nums)):

            event_number = e_nums[i]
            tree.GetEntry(event_number)
            print(event_number, sides[i])

            cell_nums = np.array(tree.tracker_cell_num)
            if with_limit: 
                if len(cell_nums) > 50: continue
            cell_nums = change_rows(cell_nums)
            bot_arr = np.array(tree.tracker_timestamp_r5)*tdc2sec
            top_arr = np.array(tree.tracker_timestamp_r6)*tdc2sec
            an_arr = np.array(tree.tracker_time_anode)
            bot_time = bot_arr - an_arr
            top_time = top_arr - an_arr
            
            for j in range(len(cell_nums)):
                if bot_time[j] > 0 and top_time[j] > 0 and an_arr[j] > 0:
                    hits_counts[all_cell_nums==cell_nums[j]] += 1
                    # print(e_nums[i])
                    # f.write(f'{cell_nums[j]}\n')
                
        for j in range(len(all_cell_nums)):
            f.write('{} {}\n'.format(all_cell_nums[j], hits_counts[j]))
            
    f.close()

def hist2D_hits(param='all', with_limit=False):

    # filename = osdir+f'hits_{param}.txt'
    filename = osdir+filedir+f'hits_cells_{param}.txt'
    if with_limit: filename = osdir+filedir+f'hits_cells_{param}_max50.txt'
    data = np.loadtxt(filename, dtype=float)
    cells, counts = data[:,0], data[:,1]
    counts[counts==0] = np.nan

    layer_ticks = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    rows = np.arange(42,56)
    layers = np.arange(-8.5,9.5)
    Z_ = counts.reshape(18, 14)
    Z = np.empty_like(Z_)
    Z[:9,:] = np.flipud(Z_[:9,:])
    Z[9:,:] = Z_[9:,:]

    fig, ax = plt.subplots(figsize=(6,6), tight_layout=True)
    title = f'Total number of hits - \'{param}\''

    c = ax.pcolormesh(rows, layers, Z, edgecolors='k', linewidths=1, cmap='magma')
    ax.plot(np.arange(41.5,56.5), np.zeros_like(np.arange(41.5,56.5)), 'k', linewidth=4) # source foil line
    fig.colorbar(c, ax=ax)
    ax.set_title(title, fontsize='large')
    ax.set_xlabel('rows')
    ax.set_ylabel('layers')
    ax.set_yticks(layers, layer_ticks)
    ax.set_xticks(rows, rows)
    ax.text(-.08, 0.75, 'France', transform=ax.transAxes, fontsize='x-large', horizontalalignment='center', verticalalignment='center', fontfamily='serif', rotation=90)
    ax.text(-.08, 0.25, 'Italy', transform=ax.transAxes, fontsize='x-large', horizontalalignment='center', verticalalignment='center', fontfamily='serif', rotation=90)
    ax.set_aspect(1)

    if with_limit:
        fig.savefig(osdir+hist3ddir+f'hist2d_hits_{param}_max50.png')
    else:
        fig.savefig(osdir+hist3ddir+f'hist2d_hits_{param}.png')

    plt.show()

def adjusted_layers(l, s):

    adjusted_layers = []

    for i in range(len(l)):
        if s[i] == 1:
            adjusted_layer =  l[i] + .5
            adjusted_layers.append(adjusted_layer)
        else:
            adjusted_layer = -l[i] - .5
            adjusted_layers.append(adjusted_layer)

    return np.array(adjusted_layers)

def hist3D_hits(param='all'):

    if param == 'fraction':

        f_all = osdir+f'lrs_all_hits_forhist_withlimit.txt'
        f_pos = osdir+f'lrs_positive_hits_forhist_withlimit.txt'

        data_all = np.loadtxt(f_all, dtype=int)
        l_all, r_all, s_all = data_all[:,0], data_all[:,1], data_all[:,2]
        l_all = adjusted_layers(l_all, s_all)
        data_pos = np.loadtxt(f_pos, dtype=int)
        l_pos, r_pos, s_pos = data_pos[:,0], data_pos[:,1], data_pos[:,2]
        l_pos = adjusted_layers(l_pos, s_pos)

        hist_all, xedges, yedges = np.histogram2d(r_all, l_all, bins=[np.arange(42,57), np.arange(-8.5,10.5)])
        hist_pos, xedges, yedges = np.histogram2d(r_pos, l_pos, bins=[np.arange(42,57), np.arange(-8.5,10.5)])
        xpos, ypos = np.meshgrid(xedges[:-1] - 0.25, yedges[:-1] - 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        hist_neg = hist_all.ravel() - hist_pos.ravel()

        dx = dy = 0.5 * np.ones_like(zpos)
        print(np.sum(hist_pos.ravel()), hist_pos.ravel()) 
        print(np.sum(hist_all.ravel()), hist_all.ravel()) 
        print(np.sum(hist_neg), hist_neg)
        # dz = hist_pos.ravel() / hist_all.ravel()

    else:
        filename = osdir+f'lrs_{param}_hits_forhist_withlimit.txt'
        data = np.loadtxt(filename, dtype=int)
        l, r, s = data[:,0], data[:,1], data[:,2]
        l = adjusted_layers(l, s)

        filename = osdir+filedir+f'hits_cells_{param}.txt'
        data = np.loadtxt(filename, dtype=int)
        cells, counts = data[:,0], data[:,1]
        counts[counts==0] = np.nan

        hist, xedges, yedges = np.histogram2d(r, l, bins=[np.arange(42,57), np.arange(-8.5,10.5)])
        xpos, ypos = np.meshgrid(xedges[:-1] - 0.25, yedges[:-1] - 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        dx = dy = 0.5 * np.ones_like(zpos)
        dz = hist.ravel()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    cmap = cm.get_cmap('magma')
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    rgba = [cmap((k-min_height)/max_height) for k in dz] 
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=rgba)
    ax.set_xlabel('Rows')
    ax.set_ylabel('Layers')
    ax.set_zlabel('Counts')

    if param == 'all':
        fig.suptitle('All hits across all cells')
        fig.savefig(osdir+hist3ddir+f'hist_all_cells.png')
    elif param == 'positive':
        fig.suptitle('All positive hits (cathode and anode) across all cells')
        fig.savefig(osdir+hist3ddir+f'hist_all_cells_pos_filter.png')
    elif param == 'negative':
        fig.suptitle('All negative hits (cathode or anode) across all cells')
        fig.savefig(osdir+hist3ddir+f'hist_all_cells_neg_filter.png')
    elif param == 'fraction':
        fig.suptitle('Fraction of positive hits (cathode or anode) compared to all registered')
        fig.savefig(osdir+hist3ddir+f'hist_all_cells_pos_fraction.png')

    plt.show()

def hist3D_hits_bis(param='all', with_limit=False):

    all_cell_nums = np.array(get_all_cell_nums())
    l, r, s = cell_id(all_cell_nums)
    l = adjusted_layers(l, s)
    filename = osdir+filedir+f'hits_cells_{param}.txt'
    if with_limit: filename = osdir+filedir+f'hits_cells_{param}_max50.txt'
    data = np.loadtxt(filename, dtype=float)
    cells, counts = data[:,0], data[:,1]
    counts[cells==1461] = 0 #np.nan
    
    # mean_resids = np.array(mean_resids)
    # bad_indices = np.isnan(mean_resids) 
    # good_indices = ~bad_indices  # remove the nan values

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(projection='3d')

    zmin = 0 #counts.min()
    zmax = counts.max()

    # xs = np.arange(41.5, 55.5, 0.1)
    # zs = np.arange(zmin, int(zmax)+0.1, 0.1)
    # X, Z = np.meshgrid(xs, zs)
    # Y = np.zeros_like(X)
    # ax.plot_surface(X, Y, Z, alpha=0.5, color='k')

    major_xticks = np.arange(42,56)
    major_yticks = np.arange(-8.5,9.5)
    layer_ticks = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    ax.set_xlim(41.5,55.5)
    ax.set_ylim(-8.5,8.5)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)
    ax.set_yticklabels(layer_ticks)
    ax.xaxis.grid(which='minor')
    ax.yaxis.grid(which='minor')

    ax.set_xlabel('Row number')
    ax.set_ylabel('Layer number')
    ax.set_zlabel('Number of hits')
    # title = 'Histogram of the mean residual value across all cells'

    x, y, z = r-.25, l-.25, zmin #np.zeros(num_bars)
    dx, dy, dz = np.ones_like(z)*1/2, np.ones_like(z)*1/2, counts - zmin

    cmap = cm.get_cmap('magma')
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    rgba = [cmap((k-min_height)/max_height) for k in dz]

    ax.bar3d(x, y, z, dx, dy, dz, zsort='average', color=rgba) 
    ax.text(42, -4, max_height, 'Italy',  fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')
    ax.text(42, +4, max_height, 'France', fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')

    ax.view_init(40,-20)

    if with_limit: 
        fig.savefig(osdir+hist3ddir+f'hist3D_hits_{param}_max50.png')
    else:
        fig.savefig(osdir+hist3ddir+f'hist3D_hits_{param}.png')

    plt.show()

def interpolate_missing_cathode_hits(event_number, side):

    filename = osdir+'mean_tdts_fromgoodtracks.txt'
    data = np.loadtxt(filename, dtype=float)
    cells, mean_tdts = data[:,0].astype(int), data[:,1]

    tree.GetEntry(event_number)

    cell_nums = np.array(tree.tracker_cell_num)
    cell_nums = change_rows(cell_nums)
    layers, rows, sides = cell_id(cell_nums)
    an_times = np.array(tree.tracker_time_anode)
    r5_arr = np.array(tree.tracker_timestamp_r5)
    r6_arr = np.array(tree.tracker_timestamp_r6)
    tc_times = r6_arr*tdc2sec
    bc_times = r5_arr*tdc2sec

    if side=='france':
        side_filter = sides==1
    else:
        side_filter = sides==0

    l = layers.copy()[side_filter]
    inds = l.argsort()

    l_sorted = layers[side_filter][inds]
    r_sorted = rows[side_filter][inds]
    s_sorted = sides[side_filter][inds]
    cells_sorted = cell_nums[side_filter][inds]
    tc_sorted = tc_times[side_filter][inds]
    bc_sorted = bc_times[side_filter][inds]
    an_sorted = an_times[side_filter][inds]
    top_sorted = (tc_sorted - an_sorted)*1e6 # microseconds
    bot_sorted = (bc_sorted - an_sorted)*1e6 # microseconds

    interpolated_cells_top = []
    interpolated_cells_bot = []

    if (tc_sorted < 0).sum() != 0: # if at least one negative
        bad_cells_top = cells_sorted[tc_sorted < 0]
        # from mean propagation time, replace the missing tdt_sorted
        for cell in bad_cells_top:
            mean_tdt = mean_tdts[cells==cell][0]
            top_drift = mean_tdt - bot_sorted[cells_sorted==cell][0]
            top_sorted[cells_sorted==cell] = top_drift # update the top drift time
            interpolated_cells_top.append(cell)

    if (bc_sorted < 0).sum() != 0: # if at least one negative
        bad_cells_bot = cells_sorted[bc_sorted < 0]
        # from mean propagation time, replace the missing tdt_sorted
        for cell in bad_cells_bot:
            mean_tdt = mean_tdts[cells==cell][0]
            bot_drift = mean_tdt - top_sorted[cells_sorted==cell][0]
            bot_sorted[cells_sorted==cell] = bot_drift # update the top drift time
            interpolated_cells_bot.append(cell)

    total_dts = top_sorted + bot_sorted
    vert_dist = vertical_fractional(top_sorted, bot_sorted)
    horz_dist = layers_to_distances(l_sorted)
    vert_errs = vertical_error(vert_dist, total_dts, cells_sorted)

    print('\n*** Event {0} ({1}) ***\n'.format(event_number, side))
    for i in range(len(l_sorted)):
        print('''cell {} / layer {} / row {} / side {} / top {:.3f} / bot {:.3f} / total {:.4f} / vertical {:.3f} +/- {:.3f}
'''.format(cells_sorted[i], l_sorted[i], r_sorted[i], s_sorted[i], top_sorted[i], bot_sorted[i], total_dts[i], vert_dist[i], vert_errs[i]))
 
    if len(interpolated_cells_top)!=0:
        print('interpolated cells, missing top cathode:', interpolated_cells_top)
    if len(interpolated_cells_bot)!=0:
        print('interpolated cells, missing bottom cathode:', interpolated_cells_bot)

    plot_vertical_fractional(vert_dist, horz_dist, vert_errs, event_number, side, interpolate=True)

def hist_r5r6_cell(cell_num, blip_bot=0, blip_top=0):

    l, r, s = cell_id(cell_num)

    bot_times, top_times, events = [], [], []

    for event in tree:

        # one event will only have one hit in a given cell, if any at all
        event_number = event.event_number
        cell_nums = np.array(event.tracker_cell_num)
        cell_nums = change_rows(cell_nums)
        if cell_num not in cell_nums:
            continue
        if len(cell_nums) > 50: # filter out certain events that have almost all of the cells triggered - 'noisy events'
            continue

        anode_time = np.array(event.tracker_time_anode)[cell_nums==cell_num][0]
        r5 = np.array(event.tracker_timestamp_r5)[cell_nums==cell_num][0]
        r6 = np.array(event.tracker_timestamp_r6)[cell_nums==cell_num][0]

        # require all three to be positive numbers
        if anode_time < 0 or r5 < 0 or r6 < 0:
            continue

        bot_time = (r5*tdc2sec - anode_time)*1e6
        top_time = (r6*tdc2sec - anode_time)*1e6

        print('event {} / bot = {:.6f} / top = {:.6f}'.format(event_number, bot_time, top_time))

        if blip_bot != 0 and bot_time < blip_bot:
            print('\nblip!\nbottom cathode\n')

        if blip_top != 0 and top_time < blip_top:
            print('\nblip!\top cathode\n')


        bot_times.append(bot_time)
        top_times.append(top_time)
        events.append(event_number)

    mean_bot, stdev_bot = np.mean(bot_times), np.std(bot_times)
    mean_top, stdev_top = np.mean(top_times), np.std(top_times)
    print('\nCell Number: {}\nMean r5 = {:.2f} / stdev = {:.2f}\nMean r6 = {:.2f} / stdev = {:.2f}\n'.format(cell_num, mean_bot, stdev_bot, mean_top, stdev_top))

    fig, ax = plt.subplots()
    # ax.plot(odd_events, odd_tdts)
    ax.hist(bot_times, bins=30, histtype='step', color='maroon')
    ax.set_xlabel('bottom cathode times')
    ax.set_ylabel('counts')
    ax.set_title('bottom cathode time distribution for cell {0} ({1}.{2}.{3})'.format(cell_num, l, r, s))
    fig.savefig(osdir+f'hist_bot_cell_{cell_num}.png')
    plt.show()

    fig, ax = plt.subplots()
    # ax.plot(odd_events, odd_tdts)
    ax.hist(top_times, bins=30, histtype='step', color='maroon')
    ax.set_xlabel('top cathode times')
    ax.set_ylabel('counts')
    ax.set_title('top cathode time distribution for cell {0} ({1}.{2}.{3})'.format(cell_num, l, r, s))
    fig.savefig(osdir+f'hist_top_cell_{cell_num}.png')
    plt.show()

def count_lines_in_file(filename):

    f = open(filename, 'r')
    print(len(f.readlines()))

def change_lines_in_file(filename):

    f = open(filename, 'r')
    newfile = open(osdir+'mean_residuals_new.txt', 'w')

    for line in f.readlines():
        print(line.split())
        for elem in line.split():
            if elem == 'np.nan': new_elem = 'nan'
            else: new_elem = elem

            newfile.write(f'{new_elem} ')
        newfile.write('\n')
    
    f.close()
    newfile.close()

def change_bb_decay_file(margin=0.1):

    filename = bb_decay_file
    new_filename = bb_decay_tracks_file
    f = open(filename, 'r')
    newfile = open(new_filename, 'w')

    for line in f.readlines():
        event_number = line.strip()
        newfile.write(f'{event_number} france\n{event_number} italy\n')

    f.close()
    newfile.close()

def get_fit_error(layers, cells, top, bot, method=1):

    vert_dist = vertical_fractional(top, bot)
    tpts = top + bot
    horz_dist = layers_to_distances(layers)
    vert_errs = vertical_error(vert_dist, tpts, cells)

    if method==1:
        popt, pcov = curve_fit(linear, horz_dist, vert_dist)
        return np.sqrt(np.diag(pcov))[0]
    if method==2:
        x = horz_dist[:,None]
        reg = LinearRegression().fit(x, vert_dist)
        return reg.score(x,vert_dist)
    if method==3:
        df = len(layers)
        fitting_params = 2 # linear regression
        popt, pcov = curve_fit(linear, horz_dist, vert_dist, sigma=vert_errs)
        resids = vert_dist - linear(horz_dist, *popt)
        chisq = np.sum((resids/vert_errs)**2)
        reduced_chisq = chisq / (df-fitting_params)
        return reduced_chisq

def goodness_of_fit_tofile():

    filename = osdir+f'tracks_with_cutoff_0.05.txt'
    filename = osdir+f'good_tracks_with_fiterror_cutoff_0.2.txt'
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    f = open(osdir+'good_tracks_with_cutoff_bestfitparams.txt', 'w')

    for i in range(len(e_nums)):

        event_number = e_nums[i]
        tree.GetEntry(event_number)
        l, r, s, top, bot = filter_a_track(event_number, sides[i])
        vert_dist = vertical_fractional(top, bot)
        horz_dist = layers_to_distances(l)
        popt, pcov = curve_fit(linear, horz_dist, vert_dist)
        perr = np.sqrt(np.diag(pcov))[0]
        print(event_number, sides[i])
        f.write(f'{event_number} {sides[i]} {popt[0]} {perr}\n')

    f.close()

def hist_goodness_of_fit():

    filename = osdir+'good_tracks_with_cutoff_bestfitparams.txt'
    data = np.loadtxt(filename, dtype=str, delimiter=', ')
    e_nums, sides, coeffs, errs = data[:,0].astype(int), data[:,1], data[:,2].astype(float), data[:,3].astype(float)

    good_inds = np.logical_and(np.isfinite(errs), errs < .4)
    bad_e_nums = e_nums[~good_inds]
    bad_sides = sides[~good_inds]

    inds = np.isfinite(errs) * (0.095 < errs) * (errs < 0.105)
    interesting_enums, interesting_sides = e_nums[inds], sides[inds]

    for i in range(len(interesting_enums)):
        print(interesting_enums[i], interesting_sides[i])

    good_errs = errs[good_inds]

    # print(e_nums[np.argmax(good_errs)], good_errs[np.argmax(good_errs)])

    mean_err = np.mean(good_errs)
    fig = plt.figure()
    plt.hist(good_errs, bins=30, color='forestgreen', histtype='step')
    plt.axvline(mean_err, color='k', linestyle='dashed', linewidth=1)
    plt.xlabel('goodness of fit')
    plt.ylabel('Counts')
    plt.title('Goodness of fit for all good tracks | mean = {:.2f}'.format(mean_err))
    fig.savefig(osdir+histdir+f'hist_goodness_of_fit.png')
    plt.show()

def get_residuals(layers, top, bot):

    vert_dist = vertical_fractional(top, bot)
    horz_dist = layers_to_distances(layers)
    popt, _ = curve_fit(linear, horz_dist, vert_dist)
    return vert_dist - linear(horz_dist, *popt) #np.abs(vert_dist - linear(horz_dist, *popt))

def write_residuals_to_file(param='good tracks'):

    if param == 'good tracks':
        filename = good_tracks_file
    elif param == 'good tracks (old)':
        filename = old_second_level_file
    elif param == 'really good tracks':
        filename = really_good_tracks
    elif param == 'bb decays 0.05':
        filename = bb_decay_tracks_file

    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    all_cell_nums = get_all_cell_nums()
    resid_mega_list = [[] for i in range(len(all_cell_nums))]
    # mean_residuals = np.zeros(len(all_cell_nums))

    for i in range(len(e_nums)):

        event_number = e_nums[i]
        tree.GetEntry(event_number)
        print(event_number, sides[i])

        l, r, s, top, bot = filter_a_track(event_number, sides[i])
        cell_nums = cell_num_(l, r, s)
        residuals = get_residuals(l, top, bot)

        for j in range(len(cell_nums)):

            ind = all_cell_nums.index(cell_nums[j])
            resid_mega_list[ind].append(residuals[j])

    # f = open(osdir+'mean_residuals_updated.txt', 'w')

    for z in range(len(all_cell_nums)):

        if len(resid_mega_list[z]) != 0:

            if param == 'good tracks':
                resid_file = osdir+f'residuals - {param}/residuals_cell_{all_cell_nums[z]}.txt'
            elif param == 'good tracks (old)':
                resid_file = osdir+f'residuals - {param}/residuals_cell_{all_cell_nums[z]}.txt'
            elif param == 'really good tracks':
                resid_file = osdir+f'residuals - {param}/residuals_cell_{all_cell_nums[z]}.txt'
            elif param == 'bb decays 0.05':
                resid_file = osdir+f'residuals - {param}/residuals_cell_{all_cell_nums[z]}.txt'
        
            np.savetxt(resid_file, resid_mega_list[z])

            mean_resid = np.mean(resid_mega_list[z])
            resid_stdev = np.std(resid_mega_list[z])
            # mean_residuals[z] = mean_resid
            print('{} {:.6f} {:.6f}\n'.format(all_cell_nums[z], mean_resid, resid_stdev))
            # f.write('{} {:.6f} {:.6f}\n'.format(all_cell_nums[z], mean_resid, resid_stdev))

        else: 
            # f.write('{} nan nan\n'.format(all_cell_nums[z])) # f.close()
            print(all_cell_nums[z], 'empty!\n')

def write_mean_std_resid_tofile(param='good tracks'):

    f = open(mean_std_residuals_file, 'w')

    all_cell_nums = get_all_cell_nums()
    for cell_num in all_cell_nums:

        if param == 'good tracks':
            resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
        elif param == 'really good tracks':
            resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
        elif param == 'bb decays':
            resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
        if Path(resid_file).exists() == False: 
            continue
        residuals = np.loadtxt(resid_file)
        mean_resid = np.mean(np.abs(residuals))
        stdev_resid = np.std(residuals)
        f.write('{} {} {}\n'.format(cell_num, mean_resid, stdev_resid))

    f.close()

def hist3D_residuals():

    data = np.loadtxt(mean_std_residuals_file, dtype=float)
    cell_nums, mean_resids, stdevs = np.array(data[:,0].astype(int)), np.array(data[:,1]), np.array(data[:,2])
    mean_resids[cell_nums==1461] = stdevs[cell_nums==1461] = np.nan # ignore the 1461 cell which behaves weirdly
    num_bars = len(cell_nums)
    l, r, s = cell_id(cell_nums)
    l = adjusted_layers(l, s)

    bad_indices = np.isnan(mean_resids) 
    good_indices = ~bad_indices  # remove the nan values

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    zmin = 0
    zmax = mean_resids[good_indices].max()

    xs = np.arange(41.5, 55.5, 0.1)
    zs = np.arange(zmin, int(zmax)+0.1, 0.1)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.5, color='k')

    major_xticks = np.arange(42,56)
    minor_xticks = np.arange(41.5,56.5)
    major_yticks = np.arange(-8.5,9.5)
    minor_yticks = np.arange(-9,10)
    layer_ticks = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    ax.set_xlim(41.5,55.5)
    ax.set_ylim(-8.5,8.5)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)
    # ax.set_xticks(minor_xticks, minor = True)
    # ax.set_yticks(minor_yticks, minor = True)
    ax.set_yticklabels(layer_ticks)
    ax.xaxis.grid(which='minor')
    ax.yaxis.grid(which='minor')

    title = 'Histogram of the mean residuals from all tracks, across all cells'

    # ax.set_zlim(0, tracker_height)
    ax.set_xlabel('Row number')
    ax.set_ylabel('Layer number')
    ax.set_zlabel('Mean residuals /m')
    ax.set_title(title)
    
    x, y, z = r[good_indices]-.25, l[good_indices]-.25, zmin #np.zeros(num_bars)
    dx, dy, dz = np.ones_like(z)*1/2, np.ones_like(z)*1/2, mean_resids[good_indices] - zmin

    # print(x,y,z,dx,dy,dz)

    cmap = cm.get_cmap('magma')
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    rgba = [cmap((k-min_height)/max_height) for k in dz]

    ax.bar3d(x, y, z, dx, dy, dz, zsort='average', color=rgba) 
    ax.text(42, -4, zmax, 'Italy',  fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')
    ax.text(42, +4, zmax, 'France', fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')

    ax.view_init(30,-25)

    # fig.suptitle(title)

    fig.savefig(osdir+hist3ddir+'hist3D_mean_resids.png')

    plt.show()

def custom_resids_hist_410(resid_cutoff = 0.20):

    filename = '/Users/casimirfisch/Desktop/Uni/SHP/Plots/Files/cell410_tracksforhist.txt'
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    resids = []

    for i in range(len(e_nums)):

        event_number = e_nums[i]
        tree.GetEntry(event_number)

        l, r, s, top, bot = filter_a_track(event_number, sides[i])
        residuals = get_residuals(l, top, bot)
        
        # if np.any(residuals > resid_cutoff):
        #     continue
        
        cell_nums = cell_num_(l,r,s)
        # if 410 not in cell_nums: 
        #     continue
    
        print(event_number, sides[i])
        
        residual = residuals[cell_nums == 410][0]
        resids.append(residual)

    resids = np.array(resids)*100
    np.savetxt(osdir+filedir+'residuals_410_cutoff_0.20.txt', resids)

    fig = plt.figure(figsize=(5,6), tight_layout=True)
    plt.hist(resids, bins=30, color='indigo', histtype='step') # range=[-30,30]
    plt.axvline(5, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(-5, color='k', linestyle='dashed', linewidth=1)
    plt.xlabel('vertical residuals /cm')
    plt.ylabel('counts')
    plt.xlim(-15,15)
    plt.text(10, 60, 'filtered out', fontsize='large', horizontalalignment='center', verticalalignment='center')
    plt.text(-10, 60, 'filtered out', fontsize='large', horizontalalignment='center', verticalalignment='center')
    plt.title('Cell 410 (row 45, layer 5, Italy)')
    fig.savefig(osdir+histdir+'hist_resids_cell_410_20cm.png')

def hist_resids_cell(cell_num, plot_or_not=True, blip=0, with_sign=False):
    
    # alter to retrieve data from files instead.

    layer, row, side = cell_id(cell_num)
    side_str = 'france' if side == 1 else 'italy'

    resids = []

    # filename = osdir+'tracks_good_all_filters_max20_min8.txt'
    filename = osdir+f'tracks_with_cutoff_0.05.txt'
    filename = first_level_file
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    for i in range(len(e_nums)):

        event_number = e_nums[i]
        tree.GetEntry(event_number)

        if sides[i] != side_str: continue
        # print(event_number, sides[i])

        l, r, s, top, bot = filter_a_track(event_number, sides[i])
        cell_nums = cell_num_(l, r, s)

        if cell_num not in cell_nums: continue
        residuals = get_residuals(l, top, bot)
        residual = residuals[cell_nums == cell_num][0]
        if with_sign==True: resids.append(residual)
        else: resids.append(np.abs(residual))

        print('Track [{} - {}] / residual = {:.4f}'.format(event_number, sides[i], residual))

        if blip != 0:
            if residual > blip:
                print('\nblip!\n')

    resids = np.array(resids)*100 #cm
    mean_resid, resid_std = np.mean(resids), np.std(resids)
    print('\nCell Number: {}\nMean residual error = {:.2f} / stdev = {:.2f}\n'.format(cell_num, mean_resid, resid_std))

    if plot_or_not==True:

        fig = plt.figure(figsize=(5,6), tight_layout=True)
        plt.hist(resids, bins=40, color='indigo', histtype='step') #range=[-6.5,6.5]
        plt.axvline(mean_resid, color='k', linestyle='dashed', linewidth=1)
        plt.xlabel('vertical residuals /cm')
        plt.ylabel('counts')

        if with_sign==True:
            fig.savefig(osdir+histdir+f'hist_resids_calc_cell_{cell_num}_withsign.png')
        else:
            fig.savefig(osdir+histdir+f'hist_resids_calc_cell_{cell_num}.png')
        plt.show()

def hist_resids_cell_fromfile(cell_num, margin=0.1, param='good tracks', with_sign=False):

    layer, row, side = cell_id(cell_num)
    side_str = 'France' if side==1 else 'Italy'

    if param == 'good tracks':
        resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
    elif param == 'good tracks (10cm)':
        resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
    elif param == 'really good tracks':
        resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
    elif param == 'bb decays':
        resid_file = osdir+f'residuals - {param} {margin}/residuals_cell_{cell_num}.txt'
    
    resids = np.loadtxt(resid_file)*100 #cm

    if with_sign==False: resids = np.abs(resids)

    mean_resid, resid_std = np.mean(resids), np.std(resids)
    print('\nCell Number: {}\nMean residual error = {:.4f} / stdev = {:.4f}\n'.format(cell_num, mean_resid, resid_std))

    fig = plt.figure(figsize=(5,6), tight_layout=True)
    plt.hist(resids, bins=30, color='indigo', histtype='step') # range=[-30,30]
    plt.axvline(5, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(-5, color='k', linestyle='dashed', linewidth=1)
    plt.xlabel('vertical residuals /cm')
    plt.ylabel('counts')
    plt.xlim(-10,10)
    plt.text(7.5, 80, 'filtered out', fontsize='large', horizontalalignment='center', verticalalignment='center')
    plt.text(-7.5, 80, 'filtered out', fontsize='large', horizontalalignment='center', verticalalignment='center')

    if with_sign:
        plt.title('Cell {} (row {}, layer {}, {})'.format(cell_num, row, layer, side_str))
        # plt.title('Vertical (fractional) residual error for cell {0} ({1}.{2}.{3})\nN = {4}'.format(cell_num, layer, row, side, len(resids)))
        fig.savefig(osdir+histdir+f'hist_resids_cell_{cell_num}_withsign_{param}.png')
    else:
        plt.title('Cell {} (row {}, layer {}, {})'.format(cell_num, row, layer, side_str))
        # plt.title('Vertical (fractional) residual error for cell {0} ({1}.{2}.{3})\nN = {4} | mean = {5:.4f}'.format(cell_num, layer, row, side, len(resids), mean_resid))
        fig.savefig(osdir+histdir+f'hist_resids_cell_{cell_num}.png')

    plt.show()

def residual_error_forallcells(param='good tracks', margin=0.05):

    all_cell_nums = get_all_cell_nums()
    resid_stds = []
    all_resids = []

    for cell_num in all_cell_nums:

        if param == 'good tracks':
            resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
        elif param == 'really good tracks':
            resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
        elif param == 'bb decays':
            resid_file = osdir+f'residuals - {param} {margin}/residuals_cell_{cell_num}.txt'

        if Path(resid_file).exists() == False:
            resid_stds.append(np.nan) 
            continue
        resids = np.loadtxt(resid_file)
        resid_stds.append(np.std(resids))
        if resids.size == 1: continue
        for resid in resids:
            all_resids.append(resid)

    all_resids = np.array(all_resids)*100 #cm
    mean_resid, resid_std = np.mean(all_resids), np.std(all_resids)
    n_bins, hist_range = 50, [-5,5]
    hist, bin_edges = np.histogram(all_resids, bins=n_bins, range=hist_range)
    bincenters = np.mean(np.vstack([bin_edges[0:-1],bin_edges[1:]]), axis=0)
    popt, pcov = curve_fit(gaussian, xdata=bincenters, ydata=hist, p0=[1000,0,10])
    sigma = np.abs(popt[1])
    print('\nsigma = {:.4f} cm\nstandard deviation = {:.4f} cm\n'.format(sigma, resid_std))

    print(mean_resid, resid_std)

    fig = plt.figure(figsize=(5,6), tight_layout=True)
    plt.hist(all_resids, bins=n_bins, histtype='step', range=hist_range, color='k', linewidth=1) #range=[-.05,.05]
    # plt.axvline(mean_resid, color='k', linestyle='dashed', linewidth=1)
    x = np.linspace(hist_range[0], hist_range[-1], 500) #np.min(all_resids), np.max(all_resids)
    # plt.plot(x, gaussian(x, *popt), 'r--')
    plt.xlabel('vertical residuals /cm')
    plt.ylabel('counts')
    plt.title('All cells')

    if param == 'good tracks':
        # plt.title('Residuals across all cells | mean = {:.4f} | stdev = {:.4f}'.format(mean_resid, resid_std))
        fig.savefig(osdir+histdir+'hist_residuals_allcells_goodtracks.png')
    elif param == 'really good tracks':
        # plt.title('Residuals across all cells (really good tracks)\nmean = {:.4f} | stdev = {:.4f}'.format(mean_resid, resid_std))
        fig.savefig(osdir+histdir+'hist_residuals_allcells_reallygood.png')
    elif param == 'bb decays':
        # plt.title('Residuals across all cells (bb decays)\nmean = {:.4f} | stdev = {:.4f}'.format(mean_resid, resid_std))
        fig.savefig(osdir+histdir+f'hist_residuals_allcells_bbdecays{margin}.png')
    plt.show()

def filter_tracks_cutoff_resid(cutoff=0.05):

    filename = osdir+'tracks_good_all_filters_max20_min8_updated2.txt'
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    newfile = osdir+f'tracks_with_cutoff_{cutoff}.txt'
    f = open(newfile, 'w')
    counter = 0

    # run through the good tracks and introduce a filter that gets rid of instances where the residuals 
    # are above a certain cutoff value (~ 0.04 or 0.05, need to justify this choice - inspection?)

    for i in range(len(e_nums)):

        event_number = e_nums[i]
        tree.GetEntry(event_number)
        # print(event_number, sides[i])

        l, r, s, top, bot = filter_a_track(event_number, sides[i])
        residuals = get_residuals(l, top, bot)
        
        if np.any(residuals > cutoff):
            continue
    
        print(event_number, sides[i])
        f.write('{} {}\n'.format(e_nums[i], sides[i]))
        counter += 1

    f.close()

    print('total tracks without cutoff:', len(e_nums))
    print('total tracks with cutoff   :', counter)

def filter_tracks_cutoff_fiterror(cutoff=0.2):

    filename = osdir+'tracks_good_all_filters_max20_min8_updated2.txt'
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    newfile = osdir+f'good_tracks_with_fiterror_cutoff_{cutoff}.txt'
    # newfile = good_tracks_file
    f = open(newfile, 'w')
    counter = 0

    # run through the good tracks and introduce a filter that gets rid of instances where the error on the fit
    # is above a certain cutoff value (0.2 by inspection of tracks).

    for i in range(len(e_nums)):

        event_number = e_nums[i]
        tree.GetEntry(event_number)
        # print(event_number, sides[i])

        l, r, s, top, bot = filter_a_track(event_number, sides[i])
        fit_err = get_fit_error(l, top, bot)
        # residuals = get_residuals(l, top, bot)
        
        if fit_err > cutoff:
            continue
    
        print(event_number, sides[i])
        f.write('{} {}\n'.format(e_nums[i], sides[i]))
        counter += 1

    f.close()

    print('total tracks without cutoff:', len(e_nums))
    print('total tracks with cutoff   :', counter)

def fit_error_onetrack(event_number, side, method=1):

    tree.GetEntry(event_number)
    l, r, s, top, bot = filter_a_track(event_number, side)
    cells = cell_num_(l, r, s)
    fit_err = get_fit_error(l, cells, top, bot, method=method)
    return fit_err

def is_a_good_track(event_number, side):

    filename = good_tracks_file
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    if event_number in e_nums:
        if side in sides[e_nums==event_number]: return True #print('good track!')
    else: return False #print('bad track!')

def check_track(event_number, side):
    if is_a_good_track(event_number, side): print('good track!')
    else: print('bad track!')

def eliminated_tracks():

    f1 = '/Users/casimirfisch/Desktop/Uni/SHP/Plots/tracks_good_all_filters_max20_min8_updated.txt'
    f2 = '/Users/casimirfisch/Desktop/Uni/SHP/Plots/tracks_good_all_filters_max20_min8.txt'

    data1 = np.loadtxt(f1, dtype=str)
    e1_nums, sides1 = data1[:,0].astype(int), data1[:,1]
    data2 = np.loadtxt(f2, dtype=str)
    e2_nums, sides2 = data2[:,0].astype(int), data2[:,1]

    for i in range(len(e2_nums)):

        if e2_nums[i] not in e1_nums:
            print(e2_nums[i], sides2[i])

def bias_in_residuals():

    all_cell_nums = np.array(get_all_cell_nums())
    l, r, s = cell_id(all_cell_nums)
    l = adjusted_layers(l, s)
    mean_resids = []

    for cell_num in all_cell_nums:

        resid_file = osdir+f'Residuals/residuals_cell_{cell_num}.txt'
        if Path(resid_file).exists() == False: 
            # print(resid_file, 'bad file')
            mean_resids.append(np.nan)
            continue
        residuals = np.loadtxt(resid_file)
        mean_resids.append(np.mean(residuals))
    
    mean_resids = np.array(mean_resids)
    bad_indices = np.isnan(mean_resids) 
    good_indices = ~bad_indices  # remove the nan values

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    zmin = 0
    zmax = mean_resids[good_indices].max()

    xs = np.arange(41.5, 55.5, 0.1)
    zs = np.arange(zmin, int(zmax)+0.1, 0.1)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.5, color='k')

    major_xticks = np.arange(42,56)
    major_yticks = np.arange(-8.5,9.5)
    layer_ticks = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    ax.set_xlim(41.5,55.5)
    ax.set_ylim(-8.5,8.5)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)
    ax.set_yticklabels(layer_ticks)
    ax.xaxis.grid(which='minor')
    ax.yaxis.grid(which='minor')

    ax.set_xlabel('Row number')
    ax.set_ylabel('Layer number')
    ax.set_zlabel('Mean residual value')
    # title = 'Histogram of the mean residual value across all cells'

    x, y, z = r[good_indices]-.25, l[good_indices]-.25, zmin #np.zeros(num_bars)
    dx, dy, dz = np.ones_like(z)*1/2, np.ones_like(z)*1/2, mean_resids[good_indices] - zmin

    cmap = cm.get_cmap('inferno')
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    rgba = [cmap((k-min_height)/max_height) for k in dz]

    ax.bar3d(x, y, z, dx, dy, dz, zsort='average', color=rgba) 
    ax.text(42, -4, max_height, 'Italy',  fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')
    ax.text(42, +4, max_height, 'France', fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')

    ax.view_init(60,-20)

    fig.savefig(osdir+hist3ddir+'hist3D_resid_bias.png')

    plt.show()

def hist3D_residuals_bis(min_resids=30, param='good tracks', margin=0.1, method='sigma'):

    all_cell_nums = np.array(get_all_cell_nums())
    l, r, s = cell_id(all_cell_nums)
    l = adjusted_layers(l, s)
    mean_resids = []
    stdev_resids = []
    sigmas = []

    for cell_num in all_cell_nums:

        layer, row, side = cell_id(cell_num)

        if param == 'good tracks':
            resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
        elif param == 'really good tracks':
            resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
        elif param == 'bb decays':
            resid_file = osdir+f'residuals - {param} {margin}/residuals_cell_{cell_num}.txt'
        if Path(resid_file).exists() == False:
            mean_resids.append(np.nan); stdev_resids.append(np.nan); sigmas.append(np.nan)
            print(f'bad cell: {cell_num} ({layer}.{row}.{side})')
            continue

        residuals = np.loadtxt(resid_file)

        if cell_num in [1434, 1435, 484]:  # unusually high residuals 1461
            mean_resids.append(np.nan); stdev_resids.append(np.nan); sigmas.append(np.nan)
            print(f'bad cell: {cell_num} ({layer}.{row}.{side}) - unusual residuals')
        elif residuals.size > min_resids: 

            sigma = get_sigma(cell_num, param, margin, plot_or_not=False)
            sigmas.append(sigma)
            mean_resids.append(np.mean(np.abs(residuals)))
            stdev_resids.append(np.std(residuals))

        else: 
            mean_resids.append(np.nan); stdev_resids.append(np.nan); sigmas.append(np.nan)
            print(f'bad cell: {cell_num} ({layer}.{row}.{side}) - {residuals.size} residuals')
    
    mean_resids, stdev_resids, sigmas = np.array(mean_resids)*100, np.array(stdev_resids)*100, np.array(sigmas)*100 #cm
    bad_indices = np.isnan(mean_resids) 
    good_indices = ~bad_indices  # remove the nan values
    overall_mean, overall_stdev = np.mean(mean_resids[good_indices]), np.mean(stdev_resids[good_indices])

    if method=='mean':
        z_arr = mean_resids
    elif method=='stdev':
        z_arr = stdev_resids
    else:
        z_arr = sigmas

    good_indices = ~np.isnan(z_arr)

    print('\nMean residual error for a cell: {:.4f}\n'.format(overall_mean))
    print('\nMean residual stdev for a cell: {:.4f}\n'.format(overall_stdev))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    zmin = 1
    zmax = z_arr[good_indices].max()

    xs = np.arange(41.5, 55.5, 0.1)
    zs = np.arange(zmin, int(zmax)+0.1, 0.1)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros_like(X)
    # ax.plot_surface(X, Y, Z, alpha=0.5, color='k')

    major_xticks = np.arange(42,56)
    major_yticks = np.arange(-8.5,9.5)
    layer_ticks = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    ax.set_xlim(41.5,55.5)
    ax.set_ylim(-8.5,8.5)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)
    ax.set_yticklabels(layer_ticks)
    ax.xaxis.grid(which='minor')
    ax.yaxis.grid(which='minor')

    ax.set_xlabel('Row number')
    ax.set_ylabel('Layer number')
    ax.set_zlabel('Mean residual value')
    # title = 'Histogram of the mean residual value across all cells'

    x, y, z = r[good_indices]-.25, l[good_indices]-.25, zmin #np.zeros(num_bars)
    dx, dy, dz = np.ones_like(z)*1/2, np.ones_like(z)*1/2, z_arr[good_indices] - zmin

    cmap = cm.get_cmap('magma')
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    rgba = [cmap((k-min_height)/max_height) for k in dz]

    ax.bar3d(x, y, z, dx, dy, dz, zsort='average', color=rgba) 
    ax.text(42, -4, zmax, 'Italy',  fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')
    ax.text(42, +4, zmax, 'France', fontsize='large', horizontalalignment='center', verticalalignment='center', fontfamily='serif')

    ax.view_init(40,-20) #  25,-15

    
    fig.savefig(osdir+hist3ddir+f'hist3D_resid_min_{min_resids}_{method}.png', transparent = True)

    plt.show()

def hist2D_cells_resids(min_resids=50, param='good tracks', margin=0.1, method='sigma'):

    all_cell_nums = np.array(get_all_cell_nums())
    l, r, s = cell_id(all_cell_nums)
    l = adjusted_layers(l, s)
    mean_resids = []
    stdev_resids = []
    sigmas = []
    number_of_resids = []
    number_lowcount_cells = 0

    for cell_num in all_cell_nums:

        layer, row, side = cell_id(cell_num)

        if param == 'good tracks':
            resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
        elif param == 'really good tracks':
            resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
        elif param == 'bb decays':
            resid_file = osdir+f'residuals - {param} {margin}/residuals_cell_{cell_num}.txt'
        if Path(resid_file).exists() == False:
            mean_resids.append(np.nan); stdev_resids.append(np.nan); sigmas.append(np.nan)
            print(f'bad cell: {cell_num} ({layer}.{row}.{side})')
            continue

        residuals = np.loadtxt(resid_file)

        if cell_num in [1434, 1435, 484]:  # unusually high residuals 1461
            mean_resids.append(np.nan); stdev_resids.append(np.nan); sigmas.append(np.nan)
            print(f'bad cell: {cell_num} ({layer}.{row}.{side}) - unusual residuals')

        elif residuals.size > min_resids: 

            sigma = get_sigma(cell_num, param, margin, plot_or_not=False)
            sigmas.append(sigma); mean_resids.append(np.mean(np.abs(residuals))); stdev_resids.append(np.std(residuals))
            # print(f'good cell: {cell_num} ({layer}.{row}.{side}) - {residuals.size} residuals')
            number_of_resids.append(residuals.size)

        else: 
            mean_resids.append(np.nan); stdev_resids.append(np.nan); sigmas.append(np.nan)
            print(f'bad cell: {cell_num} ({layer}.{row}.{side}) - {residuals.size} residuals')
            number_lowcount_cells+=1
    
    mean_resids, stdev_resids, sigmas = np.array(mean_resids)*100, np.array(stdev_resids)*100, np.array(sigmas)*100 #cm
    bad_indices = np.isnan(mean_resids) 
    good_indices = ~bad_indices  # remove the nan values
    overall_mean, overall_stdev = np.mean(mean_resids[good_indices]), np.mean(stdev_resids[good_indices])

    print('\nMean residual error for a cell: {:.4f}\n'.format(overall_mean))
    print('\nMean residual stdev for a cell: {:.4f}\n'.format(overall_stdev))

    print('Average number of resids for good cells =', np.mean(number_of_resids))
    print(f'Number of cells with a count below {min_resids} =', number_lowcount_cells)

    layer_ticks = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    rows = np.arange(42,56)
    layers = np.arange(-8.5,9.5)

    if method=='mean': 
        Z_ = mean_resids.reshape(18, 14)
    elif method=='stdev': 
        Z_ = stdev_resids.reshape(18, 14)
        ticks, ticklabels = [1.6,1.5,1.4,1.3,1.2,1.1], ['1.6cm','1.5cm','1.4cm','1.3cm','1.2cm', '1.1cm']
    else: 
        Z_ = sigmas.reshape(18, 14)

    # alter the array to plot it correctly with colormesh
    Z = np.empty_like(Z_)
    Z[:9,:] = np.flipud(Z_[:9,:])
    Z[9:,:] = Z_[9:,:]

    fig, ax = plt.subplots(figsize=(6,6), tight_layout=True)
    # title = 'Mean residual error for each cell'
    # if std==True: title = 'Mean residual stdev for each cell'

    c = ax.pcolormesh(rows, layers, Z, edgecolors='k', linewidths=1)
    ax.plot(np.arange(41.5,56.5), np.zeros_like(np.arange(41.5,56.5)), 'k', linewidth=4) # source foil line
    # fig.colorbar(c, ax=ax)
    cbar = fig.colorbar(c, ax=ax, ticks=ticks) #ticks=ticks
    cbar.ax.set_yticklabels(ticklabels)
    ax.set_xlabel('rows')
    ax.set_ylabel('layers')
    ax.set_yticks(layers, layer_ticks)
    ax.set_xticks(rows, rows)
    ax.text(-.08, 0.75, 'France', transform=ax.transAxes, fontsize='x-large', horizontalalignment='center', verticalalignment='center', fontfamily='serif', rotation=90)
    ax.text(-.08, 0.25, 'Italy', transform=ax.transAxes, fontsize='x-large', horizontalalignment='center', verticalalignment='center', fontfamily='serif', rotation=90)
    ax.set_aspect(1)

    if method=='mean':
        fig.savefig(osdir+hist3ddir+'hist2d_mean_resid.png')
    elif method=='stdev':
        fig.savefig(osdir+hist3ddir+'hist2d_stdev_resid.png')
    else:
        fig.savefig(osdir+hist3ddir+'hist2d_sigma_resid.png')

    plt.show()

def write_fit_error_tofile(method=1):

    filename = good_tracks_file #really_good_tracks
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    f = open(osdir+filedir+f'goodtracks_fiterror_method{method}.txt', 'w')

    for i in range(len(e_nums)):

        event_number, side = e_nums[i], sides[i]
        fit_err = fit_error_onetrack(event_number, side, method=method)
        print(event_number, side, fit_err)
        f.write(f'{event_number} {side} {fit_err}\n')

    f.close()

def hist_fit_error(method=1, param='goodtracks'):

    if param=='goodtracks':
        filename = osdir+filedir+f'goodtracks_fiterror_method{method}.txt'
    elif param=='reallygoodtracks':
        filename = osdir+filedir+f'reallygoodtracks_fiterror_method{method}.txt'
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides, fit_errs = data[:,0].astype(int), data[:,1], data[:,2].astype(float)

    good_inds = np.isfinite(fit_errs) #* (fit_errs < .4)
    # bad_e_nums = e_nums[~good_inds]
    # bad_sides = sides[~good_inds]
    if method==1:
        inds = np.isfinite(fit_errs) * (0.095 < fit_errs) * (fit_errs < 0.105)
    elif method==2:
        inds = np.isfinite(fit_errs) * (fit_errs < 0.1)
    elif method == 3:
        inds = fit_errs > 50
    interesting_enums, interesting_sides, interesting_fiterrs = e_nums[inds], sides[inds], fit_errs[inds]

    for i in range(len(interesting_enums)):
        print(interesting_enums[i], interesting_sides[i], interesting_fiterrs[i])

    good_errs = fit_errs[good_inds]

    # mean_err = np.mean(good_errs)
    fig = plt.figure()
    plt.hist(good_errs, bins=100, color='forestgreen', histtype='step', range=[0,50])
    # plt.axvline(mean_err, color='k', linestyle='dashed', linewidth=1)
    plt.xlabel('fit error')
    plt.ylabel('Counts')
    # plt.title('Goodness of fit for \'{}\', method {}\nmean = {:.2f}'.format(param, method, mean_err))
    fig.savefig(osdir+histdir+f'hist_fiterror_{param}_method{method}.png')
    plt.show()

def compare_fit_error_methods():

    filename = really_good_tracks
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    fit_errs_1, fit_errs_2, hits = [], [], []

    for i in range(len(e_nums)):

        event_number, side = e_nums[i], sides[i]
        tree.GetEntry(event_number)
        l, r, s, top, bot = filter_a_track(event_number, side)
        fit_err_1 = get_fit_error(l, top, bot, method=1)
        fit_err_2 = get_fit_error(l, top, bot, method=2)
        n_hits = len(l)
        fit_errs_1.append(fit_err_1)
        fit_errs_2.append(fit_err_2)
        hits.append(n_hits)
        
        print(event_number, side, fit_err_1, fit_err_2)
    
    fig, (ax1,ax2) = plt.subplots(2, sharex=True)
    ax1.scatter(hits, fit_errs_1, label='fit_err_1')
    ax1.set_ylabel('fit_err_1')
    ax2.scatter(hits, fit_errs_2, label='fit_err_2')
    ax2.set_ylabel('fit_err_2')
    plt.show()

def investigate_fit_error(param='goodtracks', method=1):

    if param=='goodtracks':
        filename = osdir+filedir+f'goodtracks_fiterror_method{method}.txt'
    elif param=='reallygoodtracks':
        filename = osdir+filedir+f'reallygoodtracks_fiterror_method{method}.txt'
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides, fit_errs = data[:,0].astype(int), data[:,1], data[:,2].astype(float)

    good_inds = np.isfinite(fit_errs) #* (fit_errs < .4)
    if method==1:
        inds = np.isfinite(fit_errs) * (0.095 < fit_errs) * (fit_errs < 0.105)
    elif method==2:
        inds = np.isfinite(fit_errs) * (fit_errs < 0.1)
    interesting_enums, interesting_sides = e_nums[inds], sides[inds]

    for i in range(len(interesting_enums)):
        print(interesting_enums[i], interesting_sides[i])

    good_errs = fit_errs[good_inds]
    good_enums, good_sides = e_nums[good_inds], sides[good_inds]

    k = 5
    inds_ordered = np.argsort(good_errs)
    worst_inds = inds_ordered[-k:] # highest values 
    best_inds = inds_ordered[:k] # lowest values
    worst_enums, worst_sides = good_enums[worst_inds], good_sides[worst_inds]
    best_enums, best_sides = good_enums[best_inds], good_sides[best_inds]

    print('worst tracks\n')
    for i in range(k):
        scan_and_plot(worst_enums[i], worst_sides[i])
    
    print('best tracks\n')
    for i in range(k):
        scan_and_plot(best_enums[i], best_sides[i])

def find_cutoff_R2():

    filename = osdir+filedir+f'goodtracks_fiterror_method2.txt'
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides, fit_errs = data[:,0].astype(int), data[:,1], data[:,2].astype(float)
    good_inds = np.isfinite(fit_errs)
    print(np.sum(~good_inds))
    fit_errs = fit_errs[good_inds]

    cutoff_trials = [0.8, 0.9, 0.95, 0.99]
    total_tracks = len(fit_errs)

    for cutoff_trial in cutoff_trials:

        valid_tracks = np.sum(fit_errs >= cutoff_trial)
        print('cutoff: {:.2f} - {:.2f}%'.format(cutoff_trial, valid_tracks/total_tracks*100))


    # a good cutoff is 0.9 -- more than 50% of the tracks are conserved, while it ensures that 
    # the bad tracks are removed.

    cutoff_trial = 0.9
    trial_valid_tracks_errs = fit_errs[fit_errs >= cutoff_trial]

    k = 1
    inds_ordered = np.argsort(trial_valid_tracks_errs)
    worst_inds = inds_ordered[-k:] # highest values 
    best_inds = inds_ordered[:k] # lowest values
    worst_enums, worst_sides = e_nums[worst_inds], sides[worst_inds]
    best_enums, best_sides = e_nums[best_inds], sides[best_inds]

    print('worst tracks\n')
    for i in range(k):
        scan_and_plot(worst_enums[i], worst_sides[i])
    
    print('best tracks\n')
    for i in range(k):
        scan_and_plot(best_enums[i], best_sides[i])

def count_noisy_events(max_hits=50):

    event_count = 0
    noisy_count = 0
    total_hits = 0
    noisy_hits = 0
    noisy_events = []
    for event in tree:
        event_count+=1
        cell_nums = list(event.tracker_cell_num)
        hits = len(cell_nums)
        total_hits+=hits
        if hits > max_hits:
            noisy_count+=1
            noisy_hits+=hits
            noisy_events.append(event.event_number)
    
    print('Noisy events:', noisy_count, '- noisy hits:', noisy_hits)
    print('Total events:', event_count, '- total hits:', total_hits)

    fig, ax = plt.subplots()
        # ax.plot(odd_events, odd_tdts)
    ax.hist(noisy_events, bins=80, histtype='step', color='teal')
    ax.set_xlabel('event number')
    ax.set_ylabel('counts')
    ax.set_title('Noisy events')
    fig.savefig(osdir+f'hist_noisyevents_max{max_hits}.png')
    plt.show()

def good_tracks_incells(cell_nums:list):

    filename = good_tracks_file
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    for i in range(len(e_nums)):

        tree.GetEntry(e_nums[i])
        l_sorted, r_sorted, s_sorted, tdt_sorted, bdt_sorted = filter_a_track(e_nums[i], sides[i])
        cells_sorted = cell_num_(l_sorted, r_sorted, s_sorted)
        for cell in cell_nums:
            if cell in cells_sorted:
                print('cell', cell, 'track', e_nums[i], sides[i])
        
def plot_50_tracks():

    filename = good_tracks_file
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    for i in range(3000,3050):
        print(e_nums[i], sides[i])
        scan_and_plot(e_nums[i], sides[i])

def gaussian(x, A, mu, sigma):
    return A * np.exp(-1/2*((x-mu)/sigma)**2)

def get_sigma(cell_num, param='good tracks', margin=0.1, plot_or_not=False):

    layer, row, side = cell_id(cell_num)
    side_str = 'France' if side==1 else 'Italy'

    if param == 'good tracks':
        resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
    elif param == 'really good tracks':
        resid_file = osdir+f'residuals - {param}/residuals_cell_{cell_num}.txt'
    elif param == 'bb decays':
        resid_file = osdir+f'residuals - {param} {margin}/residuals_cell_{cell_num}.txt'
    
    resids = np.loadtxt(resid_file)
    n_bins = 30
    hist, bin_edges = np.histogram(resids, bins=n_bins)
    bincenters = np.mean(np.vstack([bin_edges[0:-1],bin_edges[1:]]), axis=0)
    popt, pcov = curve_fit(gaussian, xdata=bincenters, ydata=hist)
    sigma = np.abs(popt[1])

    if plot_or_not == True:
        fig = plt.figure(figsize=(5,6), tight_layout=True)
        plt.hist(resids, bins=n_bins, color='k', histtype='step')
        x = np.linspace(resids.min(), resids.max(), 100)
        plt.plot(x, gaussian(x, *popt), 'r--')
        plt.xlabel('vertical residuals /m')
        plt.ylabel('counts')
        plt.show()
        fig.savefig(osdir+histdir+f'hist_gaussian_resids_cell{cell_num}.png')

    # print('Cell {}, sigma = {:.4f} cm'.format(cell_num, sigma*100))
    return sigma



# INSTEAD OF FILTERING, FIT A GAUSSIAN + BCKGD TO THE RESIDUALS TO FIND THE INHERENT 
# STANDARD DEVIATION OF THE DISTRIBUTION -- THE TRUE UNCERTAINTY OF THE VERTICAL POSITIONS.
# ISSUE MAYBE FOR THE SMALLER DISTRIBUTIONS, WITH A LOW NUMBER OF COUNT - WOULD THE FIT STILL BE GOOD THEN?

def main():

    plot_xy_event(1818)

    # second_level_filtering()

    # write_residuals_to_file()

    # write_fit_error_tofile(3)
    # hist_fit_error(method=3)

    # write_hits_cells_tofile('positive anodes', with_limit=True)
    # hist2D_hits('positive anodes', with_limit=True)

    # write_mean_tpt_to_file('good tracks')

    # write_residuals_to_file()

    # hist2D_cells_resids(method='stdev', min_resids=50)
    # hist2D_tpts()

    # custom_resids_hist_410()
    # hist3D_residuals_bis(method='stdev')
    
    # hist_resids_cell_fromfile(410, with_sign=True, param='good tracks (10cm)')

    # residual_error_forallcells()

    # hist_resids_cell_fromfile(cell_num_(0,55,0), with_sign=True)
    # hist_resids_cell_fromfile(1407, with_sign=True)
    # hist_resids_cell_fromfile(1461, with_sign=True)
    # hist_resids_cell_fromfile(1489, with_sign=True)

    # hist_tpt_fromfile_cell(cell_num_(6,49,0))
    # hist2D_tpts()
    # good_tracks_incells([1517])

    # scan_and_plot(4728, 'france')
    # scan_and_plot(4723, 'italy')
    # scan_and_plot(1642, 'italy')
    # scan_and_plot(3480, 'france')

    # 85, 4709, 4723 it, 4728 fr

    # for cell in [434,462,445,384,447,466,475,458,1458,1441,1415,1424,1460,1496,1514,1407,1489,1507,1419,1501]:
    #     hist_resids_cell_fromfile(cell, with_sign=True)

    # get_sigma(450)

    # plot_3D_2sides_events([3188,23,325,13529])

    # hist3D_residuals_bis(method='stdev')

    # plot_3D_2sides_event(14844)
    

    # hist_resids_cell_fromfile(cell_num_(7,53,0), with_sign=True)
    # hist_resids_cell_fromfile(cell_num_(6,55,1), with_sign=True)
    # hist_resids_cell_fromfile(410, with_sign=True, param='good tracks')
    # hist_resids_cell_fromfile(1461, with_sign=True)
    # hist_resids_cell_fromfile(1511, with_sign=True)

    # hist_resids_cell(410, with_sign=True)

    # write_residuals_to_file(param='good tracks (old)')

    # write_mean_std_resid_tofile()

    # scan_and_plot(480, 'france')

    # residual_error_forallcells()

    # write_residuals_to_file('good tracks')

    # residual_error_forallcells('good tracks')

    # hist2D_tpts()

    # write_mean_std_resid_tofile('good tracks')

    # plot_50_tracks()

    # residual_error_forallcells()

    # hist2D_cells_resids(method='sigma')

    # print(cell_num_(5,51,0))

    # hist2D_tpts()
    # hist2D_hits()
    # hist2D_cells_resids(std=True)

    # good_tracks_incells([400,437,1499,1510])

    # high_resid_cells = [cell_num_(5,51,0), cell_num_(0,50,0), cell_num_(3,49,1), cell_num_(8,54,1)]
    # print('cells with big residuals:')
    # for cell in high_resid_cells:
    #     print(cell)

    # scan_and_plot(5813, 'france')
    # scan_and_plot(8938, 'italy')
    # scan_and_plot(9791, 'italy')
    # scan_and_plot(10705, 'france')

    # scan_and_plot(15, 'italy')
    # plot_3D_2sides_event(11)

    # plot_3D_2sides_event(72)

    # count_lines_in_file('/Users/casimirfisch/Desktop/Uni/SHP/Plots/tracks_with_cutoff_0.05.txt')
    # count_lines_in_file(first_level_file)

    # scan_and_plot(85, 'france')


    # plot_50_tracks()

    # # count_noisy_events(40)
    # for event in [400,407,417,445,448,459,460,466,470,480,482,492,493,496,497,500]:
    #     plot_3D_2sides_event(event)

    # find_bb_decays(0.05)
    # change_bb_decay_file()

    # write_residuals_to_file('bb decays 0.05')
    # residual_error_forallcells('bb decays', 0.05)

    # scan_and_plot(2133,'italy')
    # scan_and_plot(325,'italy', invert=True)
    # scan_and_plot(325, 'france')

    # scan_and_plot(16, 'italy')

    # hist_tpt_fromfile_cell(433)

    # good_tracks_incell(433)

    # first_level_filtering()

    # plot_xy_event(15135)

    # hist3D_hits_bis('all', with_limit=True)
    # hist2D_tpts()

    # write_hits_cells_tofile('positive', with_limit=True)
    # hist2D_hits('positive')
    # hist2D_hits('positive', with_limit=True)
    # hist3D_hits_bis('positive',with_limit=True)

    # hist_tpt_fromfile_cell(399, min_val=70, max_val=85)
    # tdt_cell_all_events(cell_num_(4,46,1), plot_odd_events=True, blip=64, plot_or_not=True)


    # write_residuals_to_file()
    # write_mean_std_resid_tofile()
    # hist2D_cells_resids(std=True)

    # write_fit_error_tofile(method=1)
    # hist_fit_error(method=1, param='goodtracks')
    # find_cutoff_R2()

    # compare_fit_error_methods()

    # second_level_filtering(0.02)

    # write_mean_tpt_to_file('good tracks')
    # write_residuals_to_file(param='bb decays')

    # hist3D_hits()
    # hist3D_tdts()

    # residual_error_forallcells(param = 'bb decays')

    # tdt_cell_all_events(cell_num_(2,54,1), plot_or_not=True, blip=60, plot_odd_events=False)

    # hist3D_residuals()
    # bias_in_residuals()
    # plt.show()

    # find_bb_decays()

    # write_hits_cells_tofile('positive')
    # hist2D_hits('all')

    # goodness_of_fit_tofile()
    # hist_goodness_of_fit()

    # filter_tracks_cutoff_fiterror()
    # fit_error_onetrack(23, 'italy')
    # fit_error_onetrack(23, 'france')
    # fit_error_onetrack(23, 'italy', method=2)
    # fit_error_onetrack(23, 'france', method=2)
    # scan_and_plot(23, 'italy')
    # scan_and_plot(23, 'france')

    # hist3D_hits()
    # hist3D_tdts()
    # hist2D_tpts('good tracks')
    # print(cell_num_(3,49,1))
    # tdt_cell_all_events(cell_num_(5,55,1), plot_or_not=True, blip=60)
    # tdt_cell_good_tracks(1395)

    # layers, row, side = range(9), 52, 0
    # for l in layers:
    #     tdt_cell_all_events(cell_num_(l, row, side), plot_or_not=True)

    # hist_r5r6_cell(1450)

    # plot_xy_event(10349)
    # plot_3D_2sides_event(14476)
    # scan_and_plot(6190, 'france')
    # scan_and_plot(8649, 'france')
    # scan_track_nofilter(1800, 'france')

    # tracks = np.array([[7, 'italy'],
    #             [73, 'italy'],
    #             [117, 'france'],
    #             [191, 'italy'],
    #             [242, 'france'],
    #             [321, 'italy'],
    #             [337, 'france'],
    #             [359, 'france'],
    #             [394, 'france'],
    #             [426, 'france'],
    #             [465, 'france'],
    #             [532, 'italy'],
    #             [566, 'france'],
    #             [611, 'france'],
    #             [666, 'france']])
        
    # e_nums, sides = tracks[:,0].astype(int), tracks[:,1]

    # for i in range(len(e_nums)):
    #     plot_3D_2sides_event(e_nums[i])



    # scan_and_plot(12524, 'italy')
    # scan_and_plot(13025, 'france')
    # scan_and_plot(94, 'france')
    # scan_and_plot(129, 'italy')
    # scan_and_plot(159, 'france')
    # scan_and_plot(211, 'italy')
    # scan_and_plot(228, 'italy')
    # scan_and_plot(242, 'france')
    # scan_and_plot(305, 'france')
    # scan_and_plot(5016, 'france')
    # scan_and_plot(6161, 'france')

    # plot_3D_2sides_event(12524)
    # plot_3D_2sides_event(129)
    # plot_3D_2sides_event(228)

    # scan_and_plot(4, 'italy')
    # scan_track_nofilter(6190,'france')
    # hits_all_cells_to_file('positive')

    # hist_resids_cell(cell_num_(8,55,1), blip=50)
    # hist_resids_cell_fromfile(cell_num_(4,46,1), with_sign=False)

    # bias_in_residuals()
    # hist3D_residuals_bis(30, with_sign=False)
    # hist2D_cells_resids(30)
    # plt.show()

    # interpolate_missing_cathode_hits(55, 'italy')

    # count_lines_in_file(second_level_file)
    # change_bb_decay_file()

    # write_residuals_to_file(reallygoodtracks=True)
    # filter_tracks_cutoff_resid()

    # write_mean_tdt_to_file(param='all events')
    # write_mean_resid_tofile()

    # hist_tpt_fromfile_cell(1395, param='all events')

    # count_lines_in_file('/Users/casimirfisch/Desktop/Uni/SHP/Plots/tracks_good_all_filters_max20_min8_updated2.txt')

    # first_level_filtering()

main()
