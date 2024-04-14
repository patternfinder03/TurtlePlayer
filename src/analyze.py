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
    
    parser = argparse.ArgumentParser(description="Tabulate/Plot logs for specific session(s). Look in the log folder to get the session number you want. If you do not specify the episode number(s) it will just aggregate and tabulate all the episodes in results. Performance is used to see how it performs when exploration rate == 0(Reminder BaseAgent always has exploration rate 0; When plotting two sessions for 'performance' use the BaseAgent in first session and DQN in second to apply the DQN's intervals to base agent for comparison).")
    parser.add_argument('--type', type=str, choices=['state', 'trade', 'train', 'performance'], 
                        help='The type of analysis to generate (state, trade, or training, performance). ', required=True)
    parser.add_argument('--session1', type=int, help='The session number to plot, or the first session if comparing.', required=True)
    parser.add_argument('--session2', type=int, help='The second session number for comparison plots (optional).')
    parser.add_argument('--episode_nums1', type=list_of_ints, default='All', help='List of episode numbers to analyze. Default will analyze all episodes. Input as comma seperated list. Example: 1,2,3,4,5. Not: [1,2,3,4,5]')
    parser.add_argument('--episode_nums2', type=list_of_ints, default='All', help='(Optional)List of episode numbers to analyze. Default will analyze all episodes. Input as comma seperated list. Example: 1,2,3,4,5. Not: [1,2,3,4,5]')
    parser.add_argument('--start_date', type=int, default=None, help='(Optional)The starting date for the data range to plot. Input as YYYYMMDD. Example: 20210101')
    parser.add_argument('--end_date', type=int, default=None, help='(Optional)The ending date for the data range to plot. Input as YYYYMMDD. Example: 20210101')
    

    args = parser.parse_args()

    if args.type and args.session1:
        if args.session2:  # Plot comparison if the second session is provided
            if args.type == 'state':
                plot_state_history_comparison(args.session1, args.session2, args.episode_nums1, args.episode_nums2,args.start_date, args.end_date)
            elif args.type == 'trade':
                raise NotImplementedError("Trading log comparison is not implemented yet. Please do 1 session at a time.")
            elif args.type == 'train':
                raise NotImplementedError("Training log comparison is not implemented yet. Please do 1 session at a time.")
            elif args.type == 'performance':
                find_zero_exploration_intervals_comparison(args.session1, args.session2)
        else:  # Plot individual session logs if the second session is not provided
            if args.type == 'state':
                plot_state_history(args.session1, args.episode_nums1, args.start_date, args.end_date)
            elif args.type == 'trade':
                plot_trading_log(args.session1, args.episode_nums1, args.start_date, args.end_date)
            elif args.type == 'train':
                plot_training_results_from_log(args.session1)
            elif args.type == 'performance':
                find_zero_exploration_intervals_and_data(args.session1)
    else:
        print("Please specify the analyze type and at least one session number. Use -h for help.")
        

if __name__ == "__main__":
    main()
