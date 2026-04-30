#!/usr/bin/env python3
"""
Orbit Wars - Local Test Runner (v2.0 Enhanced)

Usage:
    python test_agent.py                                    # 1 game: main.py vs random
    python test_agent.py --agent submission.py              # specify agent file
    python test_agent.py --opponent main_v5.py              # specify opponent
    python test_agent.py --games 25                         # run multiple games
    python test_agent.py --players 4                        # 2 or 4 player games
    python test_agent.py --render                           # save first game as replay.html
    python test_agent.py --verbose                          # show turn-by-turn scores
    python test_agent.py --timeout 2.0                      # set action timeout
"""

import argparse
import sys
import time
import math
import json
import os
from datetime import datetime
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

DEFAULT_TIMEOUT = 1.0
DEFAULT_GAMES = 1
DEFAULT_PLAYERS = 2

# ============================================================
# Helper Functions
# ============================================================

def total_ships(player_id, planets, fleets):
    """Calculate total ships (on planets + in fleets) for a player."""
    ships = 0
    for p in planets:
        if p[1] == player_id:
            ships += p[5]  # index 5 = ships
    for f in fleets:
        if f[1] == player_id:
            ships += f[6]  # index 6 = ships
    return ships


def count_planets(player_id, planets):
    """Count owned planets for a player."""
    return sum(1 for p in planets if p[1] == player_id)


def get_agent_path(agent_name):
    """Resolve agent name to file path."""
    if agent_name == "random":
        return "random"
    
    # Check if it's a direct path
    if os.path.isfile(agent_name):
        return agent_name
    
    # Check common locations
    possible_paths = [
        agent_name,
        f"{agent_name}.py",
        f"agents/{agent_name}.py",
        f"submission/{agent_name}.py",
    ]
    
    for path in possible_paths:
        if os.path.isfile(path):
            return path
    
    return agent_name  # Return as-is, let kaggle_environments handle it


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ============================================================
# Main Test Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test Orbit Wars agent locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_agent.py --agent submission.py --opponent random --games 10
    python test_agent.py --agent main_v7.py --opponent main_v5.py --players 4
    python test_agent.py --render --verbose --games 5
        """
    )
    
    parser.add_argument("--agent", default="main.py",
                        help="Path to your agent file (default: main.py)")
    parser.add_argument("--opponent", default="random",
                        help="Opponent: 'random' or path to .py file (default: random)")
    parser.add_argument("--opponents", nargs="+", default=None,
                        help="Multiple opponents for round-robin testing")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help=f"Number of games to run (default: {DEFAULT_GAMES})")
    parser.add_argument("--players", type=int, default=DEFAULT_PLAYERS,
                        choices=[2, 4],
                        help=f"Number of players (default: {DEFAULT_PLAYERS})")
    parser.add_argument("--render", action="store_true",
                        help="Save HTML replay of the first game")
    parser.add_argument("--verbose", action="store_true",
                        help="Print turn-by-turn information (first game only)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"Action timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    
    args = parser.parse_args()
    
    # Resolve agent paths
    agent_path = get_agent_path(args.agent)
    opponent_paths = []
    
    if args.opponents:
        opponent_paths = [get_agent_path(opp) for opp in args.opponents]
    else:
        opponent_paths = [get_agent_path(args.opponent)]
    
    # Try to import kaggle_environments
    try:
        from kaggle_environments import make
    except ImportError:
        print("❌ ERROR: kaggle-environments not installed.")
        print("   Install with: pip install kaggle-environments")
        sys.exit(1)
    
    # Check if agent files exist
    if agent_path != "random" and not os.path.isfile(agent_path):
        print(f"❌ ERROR: Agent file not found: {agent_path}")
        sys.exit(1)
    
    for opp in opponent_paths:
        if opp != "random" and not os.path.isfile(opp):
            print(f"!  WARNING: Opponent file not found: {opp}")
    
    # Color codes
    class Colors:
        GREEN = '\033[92m'
        RED = '\033[91m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
    
    if args.no_color:
        Colors.GREEN = Colors.RED = Colors.YELLOW = Colors.BLUE = Colors.CYAN = Colors.RESET = Colors.BOLD = ""
    
    # Header
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}  ORBIT WARS - AGENT TEST SUITE v2.0{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
    print(f"\n{Colors.BOLD}Configuration:{Colors.RESET}")
    print(f"  Agent:     {Colors.GREEN}{agent_path}{Colors.RESET}")
    print(f"  Opponent:  {Colors.YELLOW}{', '.join(opponent_paths)}{Colors.RESET}")
    print(f"  Games:     {Colors.BLUE}{args.games}{Colors.RESET}")
    print(f"  Players:   {Colors.BLUE}{args.players}{Colors.RESET}")
    print(f"  Timeout:   {Colors.BLUE}{args.timeout}s{Colors.RESET}")
    if args.seed:
        print(f"  Seed:      {Colors.BLUE}{args.seed}{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}\n")
    
    # Run tests against each opponent
    all_results = []
    
    for opponent_path in opponent_paths:
        print(f"{Colors.BOLD}>  Testing vs {opponent_path}:{Colors.RESET}")
        print(f"{Colors.CYAN}{'-' * 80}{Colors.RESET}")
        
        wins, losses, draws = 0, 0, 0
        total_my_ships = 0
        total_opp_ships = 0
        total_my_planets = 0
        total_opp_planets = 0
        total_turns = 0
        total_time = 0
        game_details = []
        
        for i in range(args.games):
            t0 = time.time()
            
            # Build agent list
            if args.players == 2:
                agents = [agent_path, opponent_path]
            else:  # 4 players
                agents = [agent_path, opponent_path, "random", "random"]
            
            # Create environment
            env = make("orbit_wars", debug=True, configuration={"actTimeout": args.timeout})
            
            # Set seed if provided
            if args.seed:
                env.seed(args.seed + i)
            
            # Run game
            try:
                env.run(agents)
            except Exception as e:
                print(f"\n{Colors.RED}[X] ERROR in game {i+1}: {e}{Colors.RESET}")
                losses += 1
                continue
            
            elapsed = time.time() - t0
            total_time += elapsed
            
            # Get final state
            if len(env.steps) >= 2:
                final_obs = env.steps[-2][0]['observation']
                planets = final_obs.get('planets', [])
                fleets = final_obs.get('fleets', [])
            else:
                planets, fleets = [], []
            
            final_step = env.steps[-1]
            turns = len(env.steps) - 1
            total_turns += turns
            
            # Calculate stats for each player
            player_stats = []
            for idx in range(args.players):
                if idx < len(final_step):
                    state = final_step[idx]
                    player_id = state.observation.player
                    ships = total_ships(player_id, planets, fleets)
                    planet_count = count_planets(player_id, planets)
                    reward = state.reward
                    status = state.status
                    player_stats.append({
                        "player_id": player_id,
                        "ships": ships,
                        "planets": planet_count,
                        "reward": reward,
                        "status": status
                    })
            
            # Our stats (player 0)
            my_stats = player_stats[0] if len(player_stats) > 0 else {"ships": 0, "planets": 0}
            opp_stats = player_stats[1] if len(player_stats) > 1 else {"ships": 0, "planets": 0}
            
            my_ships = my_stats.get("ships", 0)
            opp_ships = opp_stats.get("ships", 0)
            my_planets = my_stats.get("planets", 0)
            opp_planets = opp_stats.get("planets", 0)
            
            total_my_ships += my_ships
            total_opp_ships += opp_ships
            total_my_planets += my_planets
            total_opp_planets += opp_planets
            
            # Determine result
            if my_ships > opp_ships:
                result = "WIN"
                result_color = Colors.GREEN
                wins += 1
            elif my_ships < opp_ships:
                result = "LOSS"
                result_color = Colors.RED
                losses += 1
            else:
                result = "DRAW"
                result_color = Colors.YELLOW
                draws += 1
            
            game_details.append({
                "game": i + 1,
                "my_ships": my_ships,
                "opp_ships": opp_ships,
                "my_planets": my_planets,
                "opp_planets": opp_planets,
                "turns": turns,
                "time": elapsed,
                "result": result
            })
            
            # Print game result
            status_icon = "V" if result == "WIN" else ("X" if result == "LOSS" else "~")
            print(f"  {Colors.BOLD}Game {i+1:>3}/{args.games}{Colors.RESET} | "
                  f"Ships: {Colors.BLUE}{my_ships:>6.0f}{Colors.RESET} vs {Colors.YELLOW}{opp_ships:>6.0f}{Colors.RESET} | "
                  f"Planets: {Colors.BLUE}{my_planets:>2}{Colors.RESET} vs {Colors.YELLOW}{opp_planets:>2}{Colors.RESET} | "
                  f"Diff: {result_color}{my_ships-opp_ships:>+6.0f}{Colors.RESET} | "
                  f"{turns:>3} turns | {elapsed:>5.1f}s | "
                  f"{result_color}{status_icon} {result}{Colors.RESET}")
            
            # Save replay for first game if requested
            if args.render and i == 0:
                try:
                    html = env.render(mode="html", width=900, height=650)
                    replay_file = f"replay_{Path(agent_path).stem}_vs_{Path(opponent_path).stem}.html"
                    with open(replay_file, "w", encoding="utf-8") as f:
                        f.write(html)
                    print(f"    {Colors.CYAN}> Replay saved: {replay_file}{Colors.RESET}")
                except Exception as e:
                    print(f"    {Colors.YELLOW}! Could not save replay: {e}{Colors.RESET}")
        
        # Calculate averages
        games_played = wins + losses + draws
        win_rate = 100 * wins / games_played if games_played > 0 else 0
        avg_my_ships = total_my_ships / games_played if games_played > 0 else 0
        avg_opp_ships = total_opp_ships / games_played if games_played > 0 else 0
        avg_ship_diff = avg_my_ships - avg_opp_ships
        avg_my_planets = total_my_planets / games_played if games_played > 0 else 0
        avg_opp_planets = total_opp_planets / games_played if games_played > 0 else 0
        avg_turns = total_turns / games_played if games_played > 0 else 0
        avg_time = total_time / games_played if games_played > 0 else 0
        
        # Print summary
        print(f"\n{Colors.CYAN}{'-' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}  Summary vs {opponent_path}:{Colors.RESET}")
        print(f"  {Colors.GREEN}Wins:{Colors.RESET}   {wins:>3}  |  "
              f"{Colors.RED}Losses:{Colors.RESET} {losses:>3}  |  "
              f"{Colors.YELLOW}Draws:{Colors.RESET}  {draws:>3}  |  "
              f"{Colors.BOLD}Win Rate:{Colors.RESET} {win_rate:>6.1f}%")
        print(f"  {Colors.BLUE}Avg Ships:{Colors.RESET}  Yours: {avg_my_ships:>7.1f}  |  Opponent: {avg_opp_ships:>7.1f}  |  Diff: {avg_ship_diff:>+7.1f}")
        print(f"  {Colors.BLUE}Avg Planets:{Colors.RESET} Yours: {avg_my_planets:>7.1f}  |  Opponent: {avg_opp_planets:>7.1f}")
        print(f"  {Colors.BLUE}Avg Turns:{Colors.RESET}  {avg_turns:>6.1f}  |  "
              f"{Colors.BLUE}Avg Time:{Colors.RESET} {format_time(avg_time)}")
        
        all_results.append({
            "opponent": opponent_path,
            "games": games_played,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "avg_my_ships": avg_my_ships,
            "avg_opp_ships": avg_opp_ships,
            "avg_ship_diff": avg_ship_diff,
            "avg_my_planets": avg_my_planets,
            "avg_opp_planets": avg_opp_planets,
            "avg_turns": avg_turns,
            "avg_time": avg_time,
            "game_details": game_details
        })
        
        print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}\n")
    
    # Overall summary if multiple opponents
    if len(all_results) > 1:
        total_wins = sum(r["wins"] for r in all_results)
        total_losses = sum(r["losses"] for r in all_results)
        total_draws = sum(r["draws"] for r in all_results)
        total_games = sum(r["games"] for r in all_results)
        overall_win_rate = 100 * total_wins / total_games if total_games > 0 else 0
        
        print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}  OVERALL SUMMARY (All Opponents):{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 80}{Colors.RESET}")
        print(f"  {Colors.GREEN}Total Wins:{Colors.RESET}   {total_wins:>3}  |  "
              f"{Colors.RED}Total Losses:{Colors.RESET} {total_losses:>3}  |  "
              f"{Colors.YELLOW}Total Draws:{Colors.RESET}  {total_draws:>3}")
        print(f"  {Colors.BOLD}Overall Win Rate:{Colors.RESET} {overall_win_rate:>6.1f}%  ({total_games} games)")
        print(f"{Colors.GREEN}{'=' * 80}{Colors.RESET}\n")
    
    # Save results to JSON if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_path,
            "opponents": opponent_paths,
            "configuration": {
                "games": args.games,
                "players": args.players,
                "timeout": args.timeout,
                "seed": args.seed
            },
            "results": all_results
        }
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"{Colors.CYAN}> Results saved to: {args.output}{Colors.RESET}\n")
    
    # Return exit code based on performance
    if all_results:
        avg_win_rate = sum(r["win_rate"] for r in all_results) / len(all_results)
        if avg_win_rate >= 60:
            sys.exit(0)  # Good performance
        elif avg_win_rate >= 40:
            sys.exit(1)  # Mediocre
        else:
            sys.exit(2)  # Poor performance
    else:
        sys.exit(3)  # No results


if __name__ == "__main__":
    main()
