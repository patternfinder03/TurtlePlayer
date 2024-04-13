import argparse
from gym_turtle_env.log_helpers import (plot_state_history, plot_trading_log, plot_training_results_from_log,
                         plot_state_history_comparison, find_zero_exploration_intervals_and_data,
                         find_zero_exploration_intervals_comparison)

def main():
    """
    Arg parser for plotting logs for individual sessions or comparisons between two sessions.
    """
    def list_of_ints(val):
        if val == 'All':
            return val
        elif isinstance(val, int):
            return [val]
        else:
            return [int(x) for x in val.split(',')]
    
    parser = argparse.ArgumentParser(description='Plot logs for specific session(s).')
    parser.add_argument('--plot', type=str, choices=['state', 'trade', 'train', 'performance'], 
                        help='The type of plot to generate (state, trade, or training).')
    parser.add_argument('--session1', type=int, help='The session number to plot, or the first session if comparing.')
    parser.add_argument('--session2', type=int, help='The second session number for comparison plots (optional).')
    parser.add_argument('--episode_nums1', type=list_of_ints, default='All', help='...')
    parser.add_argument('--episode_nums2', type=list_of_ints, default='All', help='...')
    parser.add_argument('--start_date', type=int, default=None, help='The starting index for the data range to plot.')
    parser.add_argument('--end_date', type=int, default=None, help='The ending index for the data range to plot.')
    

    args = parser.parse_args()

    if args.plot and args.session1:
        if args.session2:  # Plot comparison if the second session is provided
            if args.plot == 'state':
                plot_state_history_comparison(args.session1, args.session2, args.episode_nums1, args.episode_nums2,args.start_date, args.end_date)
            elif args.plot == 'trade':
                raise NotImplementedError("Trading log comparison is not implemented yet.")
            elif args.plot == 'train':
                raise NotImplementedError("Training log comparison is not implemented yet.")
            elif args.plot == 'performance':
                find_zero_exploration_intervals_comparison(args.session1, args.session2)
        else:  # Plot individual session logs if the second session is not provided
            if args.plot == 'state':
                plot_state_history(args.session1, args.episode_nums1, args.start_date, args.end_date)
            elif args.plot == 'trade':
                plot_trading_log(args.session1, args.episode_nums1, args.start_date, args.end_date)
            elif args.plot == 'train':
                plot_training_results_from_log(args.session1)
            elif args.plot == 'performance':
                find_zero_exploration_intervals_and_data(args.session1)
    else:
        print("Please specify the plot type and at least one session number. Use -h for help.")
        

if __name__ == "__main__":
    main()
