import matplotlib
matplotlib.use('TkAgg')

import ROOT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import sndisplay as sn
from scipy.optimize import curve_fit
import time

file = ROOT.TFile('/Users/casimirfisch/Desktop/Uni/SHP/red_612_output.root', 'READ')
tree = file.Get('event_tree')

'''
more work to do:

- residuals
- goodness of fit
- better estimate of vertical error bars

15142 events in total
252 = 14 rows * 18 layers
20720 'tracks' with no filter (max-min, positive_timestamps)
13161 tracks with a max=20 and min=8 filter
7559 bad tracks
188168 total hits registered across all events, of which 123861 have positive r5, r6 and r0. (~65.8%)
'''

osdir = '/Users/casimirfisch/Desktop/Uni/SHP/Plots/'
eventNum = 12
tdc2sec = 12.5e-9
tdt_error = 1 # microseconds
cell_radius = 0.04 #order of magnitude
cell_diameter = 2*cell_radius
cell_err = cell_radius
tracker_height = 3.

def filter_events(max_hits=20, min_hits=8, param='one_row'):

    # could add a filter for the number of good cathode times

    if param == 'zero_filter':

        filename = osdir+f'tracks_{param}.txt'
        f = open(filename, 'w')

        counter = 0

        for event in tree:

            event_num = event.event_number

            cell_nums = np.array(event.tracker_cell_num)
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
    cell_nums, mean_tdts = data[:,0].astype(int), data[:,1]

    errors = []

    for i in range(len(tdts)):

        mean_tdt = mean_tdts[cell_nums==cells_list[i]][0]
        # if mean_tdt == np.nan: delta_tdt = 0.01 else:
        delta_tdt = abs(tdts[i] - mean_tdt)
        # if delta_tdt == 0: delta_tdt =  None
        errors.append(delta_tdt)

    return errors

def resid_errors(cells_list):

    filename = osdir+'mean_residuals_updated.txt'
    data = np.loadtxt(filename, dtype=float)
    cell_nums, mean_resids = data[:,0].astype(int), data[:,1]

    errors = []

    for cell in cells_list:
        errors.append(mean_resids[cell_nums==cell][0])
    
    return errors

def vertical_error(vert_distances, total_dts, cells_list):

    # tdt_errs = tdt_errors(total_dts, cells_list)
    resid_errs = resid_errors(cells_list)
    # error on the vertical fractional distances

    # vert_errs = []
    # for i in range(len(tdt_errs)):
    #     if tdt_errs[i]==None: vert_errs.append(None)
    #     else: vert_errs.append(tdt_errs[i] * vert_distances[i] / total_dts[i])
    # return vert_errs

    # return tdt_errs * vert_distances / total_dts 
    return resid_errs

def vertical_distances(tc_time, bc_time):

    tracker_h = tracker_height
    total_drift_time = (tc_time + bc_time)
    fraction_top, fraction_bot = tc_time/total_drift_time, bc_time/total_drift_time
    t_distance, b_distance = fraction_top*tracker_h, fraction_bot*tracker_h

    return t_distance, b_distance

def linear(x, a, b):
    return a*x + b

def plot_vertical_fractional(vert_dist, horz_dist, vert_err, event_number, side, test=False):

    # t_dist and b_dist are mirror images of each other because they are calculated as fractions
    # only consider one of them (bottom, since the electron goes up when the bottom fraction increases)
    popt, pcov = curve_fit(linear, horz_dist, vert_dist, sigma=vert_err)
    perr = np.sqrt(np.diag(pcov))

    line_xs = np.arange(0,10)*cell_diameter
    cell_boundaries = np.arange(0.5,8.5)*cell_diameter + cell_radius

    fig, ax = plt.subplots(tight_layout=True)

    title = 'Vertical (fractional) trajectory against horizontal distance\nfor event {} - {} side'.format(event_number, side)
    ax.errorbar(horz_dist, vert_dist, yerr=vert_err, xerr=cell_err, fmt='ko')
    ax.plot(line_xs, linear(line_xs, *popt), 'r-')
    fig.suptitle(title)
    ax.set_xlabel('distance along y-axis /m')
    ax.set_ylabel('z /m')
    ax.set_xlim(line_xs[0], line_xs[-1])

    for boundary in cell_boundaries:
        ax.axvline(boundary, color='k', linestyle='dashed', alpha=0.5, linewidth=1)

    ax.text(0.05, 0.05, 'error of fit: {:.4e}'.format(perr[0]), transform=ax.transAxes)

    if test==False:
        fig.savefig(osdir+'frac_vertical_dist_event{0}_{1}.png'.format(event_number, side))
    else:
        fig.savefig(osdir+'frac_vertical_dist_event{0}_{1}_test.png'.format(event_number, side))

    plt.show()

def filter_event(event_number):

    tree.GetEntry(event_number)

    cell_nums = np.array(tree.tracker_cell_num)
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

    positive_filter = np.logical_and.reduce((tc_times[side_filter]>0, bc_times[side_filter]>0, an_times[side_filter]>0))
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
    title = f'3D tracks [{filename}]'

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

    fig.suptitle(title)
    fig.savefig(osdir+'3D_events_{}.png'.format(filename))
    plt.show()

def plot_3D_2sides_event(event_number, scatter=True):

    fig = plt.figure()
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
    title = f'3D plot of event {event_number}'

    l_sorted, r_sorted, s_sorted, tdt_sorted, bdt_sorted = filter_event(event_number)
    vert_dist = vertical_fractional(tdt_sorted, bdt_sorted)
    l_sorted = adjusted_layers(l_sorted, s_sorted)

    if scatter == True:
        ax.scatter(r_sorted, l_sorted, vert_dist, color='r')
    else:
        ax.plot(r_sorted, l_sorted, vert_dist)

    ax.view_init(20,-20)
    fig.suptitle(title)
    fig.savefig(osdir+'3D_2sides_event{}.png'.format(event_number))
    plt.show()

def scan_track_nofilter(event_number, side):

    tree.GetEntry(event_number)

    cell_nums = np.array(tree.tracker_cell_num)
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

def scan_and_plot(event_number, side='france'):

    if is_a_good_track(event_number, side): print('good track!')
    else: print('bad track!')

    l_sorted, r_sorted, s_sorted, tdt_sorted, bdt_sorted = filter_a_track(event_number, side)
    cells_sorted = cell_num_(l_sorted, r_sorted, s_sorted)
    total_dts = tdt_sorted + bdt_sorted
    vert_dist = vertical_fractional(tdt_sorted, bdt_sorted)
    horz_dist = layers_to_distances(l_sorted)
    vert_errs = vertical_error(vert_dist, total_dts, cells_sorted)

    # t_distance, b_distance = vertical_distances(tdt_sorted, bdt_sorted)
    # b_dist_avg, err = vertical_dist_err(tdt_sorted, bdt_sorted)

    print('\n*** Event {0} ({1}) ***\n'.format(event_number, side))
    for i in range(len(l_sorted)):
        print('''layer {} / row {} / side {} / top {:.3f} / bot {:.3f} / total {:.4f} / vertical {:.3f} +/- {:.3f}
'''.format(l_sorted[i], r_sorted[i], s_sorted[i], tdt_sorted[i], bdt_sorted[i], total_dts[i], vert_dist[i], vert_errs[i]))

    # plot_vertical(t_distance, b_distance, l_sorted, event_number, side)
    # plot_vertical_avg_err(b_dist_avg, err, l_sorted, event_number, side)
    # plot_3D(b_dist_avg, l_sorted, r_sorted, event_number, side)
    plot_vertical_fractional(vert_dist, horz_dist, vert_errs, event_number, side)

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
    layers, rows, sides = cell_id(cell_nums)
    layers = adjusted_layers(layers, sides)
    an_times = np.array(tree.tracker_time_anode)
    r5_arr = np.array(tree.tracker_timestamp_r5)
    r6_arr = np.array(tree.tracker_timestamp_r6)

    cath_filter = np.logical_and(r5_arr>0, r6_arr>0)
    positive_filter = np.logical_and(cath_filter, an_times>0)
    negative_filter = np.invert(positive_filter)

    fig, ax = plt.subplots(figsize=(8,8))

    major_xticks = np.arange(42,56)
    minor_xticks = np.arange(41.5,56.5)
    major_yticks = np.arange(-8.5,9.5)
    minor_yticks = np.arange(-9,10)
    layer_ticks = [8,7,6,5,4,3,2,1,0,0,1,2,3,4,5,6,7,8]

    good_label = 'good hits'
    bad_label = 'hits filtered out'
    ax.scatter(rows[positive_filter], layers[positive_filter], s=200, marker='o', color='g', alpha=0.6, label=good_label)
    ax.scatter(rows[negative_filter], layers[negative_filter], s=200, marker='o', color='r', alpha=0.6, label=bad_label)
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
    ax.text(0.5, 0.75, 'France', transform=ax.transAxes, fontsize='xx-large', horizontalalignment='center', verticalalignment='center', fontfamily='serif', fontweight='bold')
    ax.text(0.5, 0.25, 'Italy', transform=ax.transAxes, fontsize='xx-large', horizontalalignment='center', verticalalignment='center', fontfamily='serif', fontweight='bold')
    ax.legend()

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

def tdt_cell_all_events(cell_num, plot_or_not=False, blip=0, plot_odd_events=False):

    l, r, s = cell_id(cell_num)

    count_neghits = 0

    tdts, events = [], []
    if plot_odd_events: 
        odd_events, odd_tdts = [], []

    for event in tree:

        # one event will only have one hit in a given cell, if any at all
        event_number = event.event_number
        cell_nums = np.array(event.tracker_cell_num)
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
            continue

        total_drift_time = (top_cath_time+bot_cath_time)*1e6

        if  blip != 0: # to look for outliers
            print('Event {} / tdt: {:.3f}'.format(event_number, total_drift_time))
            if total_drift_time < blip:
                print('blip!')
                if plot_odd_events: 
                    odd_events.append(event_number)
                    odd_tdts.append(total_drift_time)
        # if total_drift_time > 50:
        tdts.append(total_drift_time)
        events.append(event_number)

    print('negative hits:', count_neghits)

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

def write_mean_tdt_to_file(param='all events'):

    if param == 'all events':

        all_cell_nums = get_all_cell_nums()
        tdt_mega_list = [[] for i in range(len(all_cell_nums))]
        mean_tdts = np.zeros(len(all_cell_nums))

        for event in tree:

            print(event.event_number)
            cell_nums = np.array(event.tracker_cell_num)

            if 8 > len(cell_nums) or len(cell_nums) > 50: 
            # filter out certain events that have almost all of the cells triggered
                continue

            anode_times = np.array(event.tracker_time_anode)
            r5_arr = np.array(event.tracker_timestamp_r5)
            r6_arr = np.array(event.tracker_timestamp_r6)

            for i in range(len(anode_times)):

                if anode_times[i] < 0 or r5_arr[i] < 0 or r6_arr[i] < 0:
                    continue

                tdt = (tdc2sec*(r5_arr[i] + r6_arr[i]) - 2*anode_times[i])*1e6
                ind = all_cell_nums.index(cell_nums[i])
                tdt_mega_list[ind].append(tdt)

        f = open(osdir+'mean_tdts.txt', 'w')

        for j in range(len(all_cell_nums)):
            mean_tdt = np.mean(tdt_mega_list[j])
            tdt_stdev = np.std(tdt_mega_list[j])
            mean_tdts[j] = mean_tdt
            f.write('{} {:.2f} {:.2f}\n'.format(all_cell_nums[j], mean_tdt, tdt_stdev))

        f.close()
    
    elif param == 'good tracks only':

        filename = osdir+f'tracks_with_cutoff_0.05.txt'
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

            tdt = top + bot
            ind = all_cell_nums.index(cell_nums[i])
            tdt_mega_list[ind].append(tdt)

        f = open(osdir+'mean_tdts_fromgoodtracks.txt', 'w')

        for j in range(len(all_cell_nums)):
            mean_tdt = np.mean(tdt_mega_list[j])
            tdt_stdev = np.std(tdt_mega_list[j])
            mean_tdts[j] = mean_tdt
            f.write('{} {:.2f} {:.2f}\n'.format(all_cell_nums[j], mean_tdt, tdt_stdev))

        f.close()

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

    fig.savefig(osdir+'hist3D_mean_tdts.png')

    plt.show()

def hits_all_cells_to_file(param='all'):

    # function for number of hits total for all cells  
    # all valid cells (with both cathode times)? or all cells generally

    filename = osdir+f'lrs_{param}_hits_forhist_withlimit.txt'

    f=open(filename, 'w')
     
    for event in tree:

        print(event.event_number)
        cell_nums = np.array(event.tracker_cell_num)
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
        fig.savefig(osdir+f'hist_all_cells.png')
    elif param == 'positive':
        fig.suptitle('All positive hits (cathode and anode) across all cells')
        fig.savefig(osdir+f'hist_all_cells_pos_filter.png')
    elif param == 'negative':
        fig.suptitle('All negative hits (cathode or anode) across all cells')
        fig.savefig(osdir+f'hist_all_cells_neg_filter.png')
    elif param == 'fraction':
        fig.suptitle('Fraction of positive hits (cathode or anode) compared to all registered')
        fig.savefig(osdir+f'hist_all_cells_pos_fraction.png')

    plt.show()

def interpolate_missing_cathode_hits(event_number, side):

    filename = osdir+'mean_tdts.txt'
    data = np.loadtxt(filename, dtype=float)
    cells, mean_tdts = data[:,0].astype(int), data[:,1]

    tree.GetEntry(event_number)

    cell_nums = np.array(tree.tracker_cell_num)
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

    if (tc_sorted < 0).sum() != 0: # if at least one negative
        bad_cells_top = cells_sorted[tc_sorted < 0]
        # from mean propagation time, replace the missing tdt_sorted
        for cell in bad_cells_top:
            mean_tdt = mean_tdts[cells==cell][0]
            top_drift = mean_tdt - bot_sorted[cells_sorted==cell][0]
            top_sorted[cells_sorted==cell] = top_drift # update the top drift time

    if (bc_sorted < 0).sum() != 0: # if at least one negative
        bad_cells_bot = cells_sorted[bc_sorted < 0]
        # from mean propagation time, replace the missing tdt_sorted
        for cell in bad_cells_bot:
            mean_tdt = mean_tdts[cells==cell][0]
            bot_drift = mean_tdt - top_sorted[cells_sorted==cell][0]
            bot_sorted[cells_sorted==cell] = bot_drift # update the top drift time

    total_dts = top_sorted + bot_sorted
    vert_dist = vertical_fractional(top_sorted, bot_sorted)
    horz_dist = layers_to_distances(l_sorted)
    vert_errs = vertical_error(vert_dist, total_dts, cells_sorted)
    plot_vertical_fractional(vert_dist, horz_dist, vert_errs, event_number, side, test=True)

def hist_r5r6_cell(cell_num, blip_bot=0, blip_top=0):

    l, r, s = cell_id(cell_num)

    bot_times, top_times, events = [], [], []

    for event in tree:

        # one event will only have one hit in a given cell, if any at all
        event_number = event.event_number
        cell_nums = np.array(event.tracker_cell_num)
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

def get_residuals(layers, top, bot):

    vert_dist = vertical_fractional(top, bot)
    horz_dist = layers_to_distances(layers)
    popt, _ = curve_fit(linear, horz_dist, vert_dist)
    return np.abs(vert_dist - linear(horz_dist, *popt))

def write_residuals_to_file(cutoff):

    filename = osdir+'tracks_good_all_filters_max20_min8.txt'
    filename = osdir+f'tracks_with_cutoff_{cutoff}.txt'
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    all_cell_nums = get_all_cell_nums()
    resid_mega_list = [[] for i in range(len(all_cell_nums))]
    mean_residuals = np.zeros(len(all_cell_nums))

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

    f = open(osdir+'mean_residuals_updated.txt', 'w')

    for z in range(len(all_cell_nums)):

        if len(resid_mega_list[z]) != 0:

            mean_resid = np.mean(resid_mega_list[z])
            resid_stdev = np.std(resid_mega_list[z])
            mean_residuals[z] = mean_resid
            print('{} {:.6f} {:.6f}\n'.format(all_cell_nums[z], mean_resid, resid_stdev))
            f.write('{} {:.6f} {:.6f}\n'.format(all_cell_nums[z], mean_resid, resid_stdev))

        else: 

            print(all_cell_nums[z], 'empty!\n')
            f.write('{} nan nan\n'.format(all_cell_nums[z]))

    f.close()

def hist3D_residuals():

    data = np.loadtxt(osdir+'mean_residuals.txt', dtype=float)
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

    # ax.set_zlim(0, tracker_height)
    ax.set_xlabel('Row number')
    ax.set_ylabel('Layer number')
    ax.set_zlabel('Mean residuals')
    title = 'Histogram of the mean residuals from all tracks, across all cells'

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

    ax.view_init(60,-20)

    fig.savefig(osdir+'hist3D_mean_resids.png')

    plt.show()

def hist_resids_cell(cell_num, plot_or_not=True, blip=0):

    layer, row, side = cell_id(cell_num)
    side_str = 'france' if side == 1 else 'italy'

    resids = []

    filename = osdir+'tracks_good_all_filters_max20_min8.txt'
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
        resids.append(residual)

        print('Track [{} - {}] / residual = {:.4f}'.format(event_number, sides[i], residual))

        if blip != 0:
            if residual > blip:
                print('\nblip!\n')

    mean_resid, resid_std = np.mean(resids), np.std(resids)
    print('\nCell Number: {}\nMean residual error = {:.2f} / stdev = {:.2f}\n'.format(cell_num, mean_resid, resid_std))

    if plot_or_not==True:

        fig = plt.figure()
        plt.hist(resids, bins=30, color='darkmagenta', histtype='step')
        plt.axvline(mean_resid, color='k', linestyle='dashed', linewidth=1)
        plt.xlabel('vertical distance residual')
        plt.ylabel('Counts')
        plt.title('Vertical (fractional) residual error for cell {0} ({1}.{2}.{3})\nN = {4} | mean = {5:.2f}'.format(cell_num, layer, row, side, len(resids), mean_resid))
        fig.savefig(osdir+f'hist_resids_cell_{cell_num}.png')
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

def is_a_good_track(event_number, side):

    filename = osdir+f'tracks_with_cutoff_0.05.txt'
    data = np.loadtxt(filename, dtype=str)
    e_nums, sides = data[:,0].astype(int), data[:,1]

    if event_number in e_nums:
        if sides[e_nums==event_number][0] == side: return True #print('good track!')
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


def main():

    # check_track(4, 'italy')
    # filter_events(param='good_all_filters')

    # eliminated_tracks()

    # hist3D_hits()
    # hist3D_tdts()

    # tdt_cell_all_events(cell_num_(2,54,1), plot_or_not=True, blip=60, plot_odd_events=False)

    # change_lines_in_file(osdir+'mean_residuals.txt')

    # hist3D_residuals()
    # print(cell_num_(3,49,1))
    # tdt_cell_all_events(cell_num_(3,49,1), plot_or_not=True, blip=50)


    # layers, row, side = range(9), 52, 0
    # for l in layers:
    #     tdt_cell_all_events(cell_num_(l, row, side), plot_or_not=True)

    # hist_r5r6_cell(1450)

    # plot_xy_event(1339)
    # plot_3D_2sides_event(11712)
    # scan_and_plot(3707, 'france')
    # scan_and_plot(8649, 'france')
    # scan_track_nofilter(1800, 'france')

    # scan_and_plot(6, 'france')
    # scan_and_plot(9548, 'france')
    # scan_and_plot(5321, 'france')
    # scan_and_plot(5016, 'france')
    # scan_and_plot(6161, 'france')

    # scan_and_plot(4, 'italy')
    # scan_track_nofilter(6190,'france')
    # hits_all_cells_to_file('positive')

    # hist_resids_cell(cell_num_(3,49,1), blip=0)

    # interpolate_missing_cathode_hits(4, 'italy')

    # write_residuals_to_file(cutoff=0.05)
    filter_tracks_cutoff_resid()

    # count_lines_in_file('/Users/casimirfisch/Desktop/Uni/SHP/Plots/tracks_good_all_filters_max20_min8_updated2.txt')


main()
