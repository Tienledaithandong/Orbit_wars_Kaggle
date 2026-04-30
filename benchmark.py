#!/usr/bin/env python3
"""
Orbit Wars – Full Round-Robin Benchmark (v2.0 Enhanced)

Runs all agents against each other multiple times and produces a ranking.

Usage:
    python benchmark.py                    # Default: 25 games, sequential
    python benchmark.py --games 50         # 50 games per matchup
    python benchmark.py --parallel         # Run matches in parallel
    python benchmark.py --players 4        # 4-player games
    python benchmark.py --timeout 2.0      # 2 second action timeout
    python benchmark.py --output results.json  # Save to JSON
    python benchmark.py --generate-bash    # Generate bash script to run tests
"""

import subprocess
import sys
import re
import itertools
import argparse
import json
import time
import os
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

# Add your agents here
# The 'ulti' agent is the strongest based on main_ulti1.py
AGENTS = {
    "ulti": "main_ulti.py",
    "random": "random"  # Built-in random agent for baseline comparison
}

INCLUDE_RANDOM = True
DEFAULT_GAMES = 25
DEFAULT_PLAYERS = 2
DEFAULT_TIMEOUT = 1.0

# ============================================================
# Match Runner
# ============================================================

def run_match(agent1: str, agent2: str, games: int, players: int = 2, timeout: float = 1.0):
    """Run 'games' matches between agent1 and agent2. Returns dict with stats."""
    cmd = [
        sys.executable, "test_agent.py",
        "--agent", agent1,
        "--opponent", agent2,
        "--games", str(games),
        "--players", str(players),
        "--timeout", str(timeout),
        "--no-color"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=games * 30)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {
            "wins": 0, "losses": games, "draws": 0,
            "win_rate": 0.0, "avg_reward": 0.0,
            "my_ships_avg": 0.0, "opp_ships_avg": 0.0,
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "wins": 0, "losses": games, "draws": 0,
            "win_rate": 0.0, "avg_reward": 0.0,
            "my_ships_avg": 0.0, "opp_ships_avg": 0.0,
            "error": str(e)
        }

    wins = losses = draws = 0
    avg_reward = 0.0
    my_ships_avg = 0.0
    opp_ships_avg = 0.0

    # Parse the summary lines
    for line in output.splitlines():
        if "Wins:" in line or "Total Wins:" in line:
            # Parse: "Wins:   15  |  Losses:  8  |  Draws:   2  |  Win Rate:  60.0%"
            w_match = re.search(r"Wins:\s*(\d+)", line)
            l_match = re.search(r"Losses:\s*(\d+)", line)
            d_match = re.search(r"Draws:\s*(\d+)", line)
            if w_match:
                wins = int(w_match.group(1))
            if l_match:
                losses = int(l_match.group(1))
            if d_match:
                draws = int(d_match.group(1))
        
        if "Win Rate:" in line:
            wr_match = re.search(r"Win Rate:\s*([\d.]+)%", line)
            if wr_match:
                pass  # We calculate from wins/losses/draws
        
        if "Avg Ships:" in line or "Average ships:" in line:
            # Parse: "Avg Ships:  Yours:   123.4  |  Opponent:    98.7  |  Diff:   +24.7"
            y_match = re.search(r"Yours:\s*([\d.]+)", line)
            o_match = re.search(r"Opponent:\s*([\d.]+)", line)
            if y_match:
                my_ships_avg = float(y_match.group(1))
            if o_match:
                opp_ships_avg = float(o_match.group(1))

    total = wins + losses + draws
    win_rate = wins / total if total > 0 else 0.0
    
    return {
        "wins": wins, "losses": losses, "draws": draws,
        "win_rate": win_rate,
        "avg_reward": 0.0,  # Not tracked in new format
        "my_ships_avg": my_ships_avg,
        "opp_ships_avg": opp_ships_avg,
    }


def run_match_worker(args):
    """Wrapper for parallel execution."""
    return run_match(*args)


# ============================================================
# Elo Calculation
# ============================================================

def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a, rating_b, score_a, k=32):
    expected_a = expected_score(rating_a, rating_b)
    return rating_a + k * (score_a - expected_a)


# ============================================================
# Tournament Runner
# ============================================================

class TournamentRunner:
    def __init__(self, agents, games=25, players=2, timeout=1.0, include_random=True, parallel=False):
        self.agents = dict(agents)
        self.games = games
        self.players = players
        self.timeout = timeout
        self.include_random = include_random
        self.parallel = parallel
        self.results = defaultdict(lambda: defaultdict(dict))
        self.start_time = None
        
        if include_random:
            self.agents["random"] = "random"
    
    def run(self):
        """Run the full tournament."""
        self.start_time = time.time()
        
        # Print header
        print("\n" + "=" * 90)
        print("  ORBIT WARS – TOURNAMENT BENCHMARK v2.0")
        print("=" * 90)
        print(f"\n  Agents: {', '.join(self.agents.keys())}")
        print(f"  Games per matchup: {self.games}")
        print(f"  Players: {self.players}")
        print(f"  Timeout: {self.timeout}s")
        print(f"  Parallel: {'Yes' if self.parallel else 'No'}")
        print("=" * 90 + "\n")
        
        # Generate all matchups (home & away)
        matchups = []
        agent_items = list(self.agents.items())
        for (name1, file1), (name2, file2) in itertools.combinations(agent_items, 2):
            matchups.append((name1, file1, name2, file2))
            matchups.append((name2, file2, name1, file1))
        
        total_matches = len(matchups)
        print(f"  Total matchups: {total_matches}")
        print(f"  Estimated time: ~{total_matches * self.games * 2 / 60:.1f} minutes\n")
        print("-" * 90)
        
        # Run matches
        if self.parallel:
            self._run_parallel(matchups)
        else:
            self._run_sequential(matchups)
        
        # Print results
        self._print_summary()
        self._print_matrix()
        
        return self.results
    
    def _run_sequential(self, matchups):
        """Run matches sequentially."""
        for i, (name1, file1, name2, file2) in enumerate(matchups, 1):
            print(f"[{i:3d}/{len(matchups)}] {name1:12} vs {name2:12} ... ", end="", flush=True)
            
            stats = run_match(file1, file2, self.games, self.players, self.timeout)
            self.results[name1][name2] = stats
            
            wr = stats['win_rate'] * 100
            print(f"{stats['wins']}W/{stats['losses']}L/{stats['draws']}D ({wr:5.1f}%)")
    
    def _run_parallel(self, matchups):
        """Run matches in parallel."""
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor for better compatibility
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(run_match_worker, (file1, file2, self.games, self.players, self.timeout)): (name1, name2)
                for name1, file1, name2, file2 in matchups
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                name1, name2 = futures[future]
                try:
                    stats = future.result()
                    self.results[name1][name2] = stats
                    
                    wr = stats['win_rate'] * 100
                    print(f"[{i:3d}/{len(matchups)}] {name1:12} vs {name2:12}: "
                          f"{stats['wins']}W/{stats['losses']}L/{stats['draws']}D ({wr:5.1f}%)")
                except Exception as e:
                    print(f"[{i:3d}/{len(matchups)}] {name1:12} vs {name2:12}: ERROR - {e}")
    
    def _calculate_stats(self):
        """Calculate aggregate statistics for each agent."""
        agent_names = list(self.agents.keys())
        
        stats = {name: {
            "total_wins": 0, "total_losses": 0, "total_draws": 0,
            "total_games": 0, "total_ship_diff": 0.0, "match_count": 0
        } for name in agent_names}
        
        for a in agent_names:
            for b in agent_names:
                if a == b:
                    continue
                if b in self.results[a]:
                    # When a is Player 0
                    r = self.results[a][b]
                    stats[a]["total_wins"] += r["wins"]
                    stats[a]["total_losses"] += r["losses"]
                    stats[a]["total_draws"] += r["draws"]
                    stats[a]["total_games"] += r["wins"] + r["losses"] + r["draws"]
                    stats[a]["total_ship_diff"] += r["my_ships_avg"] - r["opp_ships_avg"]
                    stats[a]["match_count"] += 1
                
                if a in self.results[b]:
                    # When a is Player 1 (b is Player 0)
                    r = self.results[b][a]
                    stats[a]["total_wins"] += r["losses"]  # a wins when b loses
                    stats[a]["total_losses"] += r["wins"]  # a loses when b wins
                    stats[a]["total_draws"] += r["draws"]
                    stats[a]["total_games"] += r["wins"] + r["losses"] + r["draws"]
                    stats[a]["total_ship_diff"] += r["opp_ships_avg"] - r["my_ships_avg"]
                    # Do not increment match_count again to avoid double counting the matchup pair
        
        # Calculate derived stats
        for name in agent_names:
            s = stats[name]
            s["win_rate"] = s["total_wins"] / s["total_games"] if s["total_games"] > 0 else 0
            s["avg_ship_diff"] = s["total_ship_diff"] / s["match_count"] if s["match_count"] > 0 else 0
        
        return stats
    
    def _calculate_elo(self):
        """Calculate Elo ratings for each agent."""
        agent_names = list(self.agents.keys())
        elo = {name: 1200.0 for name in agent_names}
        
        for a in agent_names:
            for b in agent_names:
                if a >= b or b not in self.results[a]:
                    continue
                
                r_ab = self.results[a][b]
                r_ba = self.results[b][a] if a in self.results[b] else None
                
                games_ab = r_ab["wins"] + r_ab["losses"] + r_ab["draws"]
                games_ba = r_ba["wins"] + r_ba["losses"] + r_ba["draws"] if r_ba else 0
                total = games_ab + games_ba
                
                if total == 0:
                    continue
                
                score_a = r_ab["wins"] + 0.5 * r_ab["draws"]
                if r_ba:
                    score_a += r_ba["losses"] + 0.5 * r_ba["draws"]
                
                expected_a = expected_score(elo[a], elo[b])
                elo[a] = update_elo(elo[a], elo[b], score_a / total)
                elo[b] = update_elo(elo[b], elo[a], 1 - score_a / total)
        
        return elo
    
    def _print_summary(self):
        """Print final ranking summary."""
        stats = self._calculate_stats()
        elo = self._calculate_elo()
        agent_names = list(self.agents.keys())
        
        # Sort by win rate, then Elo
        sorted_agents = sorted(agent_names, key=lambda x: (stats[x]["win_rate"], elo[x]), reverse=True)
        
        elapsed = time.time() - self.start_time
        
        print("\n" + "=" * 90)
        print(f"  TOURNAMENT COMPLETE ({elapsed/60:.1f} minutes)")
        print("=" * 90)
        print(f"\n  {'Rank':<5} {'Agent':<14} {'Win Rate':<10} {'Elo':<8} {'W/L/D':<14} {'Avg Ship Diff':<14}")
        print("  " + "-" * 86)
        
        for rank, name in enumerate(sorted_agents, 1):
            s = stats[name]
            e = int(round(elo[name]))
            wld = f"{s['total_wins']}/{s['total_losses']}/{s['total_draws']}"
            diff = s["avg_ship_diff"]
            
            rank_icon = "1." if rank == 1 else ("2." if rank == 2 else ("3." if rank == 3 else f"{rank}."))
            
            print(f"  {rank_icon:<5} {name:<14} {s['win_rate']*100:>6.1f}%    {e:<8} {wld:<14} {diff:>+8.1f}")
        
        print("=" * 90)
    
    def _print_matrix(self):
        """Print head-to-head win rate matrix."""
        stats = self._calculate_stats()
        agent_names = list(self.agents.keys())
        sorted_agents = sorted(agent_names, key=lambda x: stats[x]["win_rate"], reverse=True)
        
        print("\n" + "=" * 90)
        print("  HEAD-TO-HEAD WIN RATE MATRIX (row vs column)")
        print("=" * 90)
        
        # Header
        header = f"  {'':<12}"
        for name in sorted_agents:
            header += f"{name:>8}"
        print(header)
        print("  " + "-" * (12 + len(sorted_agents) * 8))
        
        # Rows
        for a in sorted_agents:
            row = f"  {a:<12}"
            for b in sorted_agents:
                if a == b:
                    row += f"{'---':>8}"
                elif b in self.results[a]:
                    wr = self.results[a][b]["win_rate"] * 100
                    row += f"{wr:>7.1f}%"
                else:
                    row += f"{'N/A':>8}"
            print(row)
        
        print("=" * 90 + "\n")
    
    def save_results(self, filename):
        """Save results to JSON file."""
        stats = self._calculate_stats()
        elo = self._calculate_elo()
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "agents": list(self.agents.keys()),
                "games": self.games,
                "players": self.players,
                "timeout": self.timeout,
                "parallel": self.parallel
            },
            "rankings": [
                {
                    "rank": i + 1,
                    "agent": name,
                    "win_rate": stats[name]["win_rate"],
                    "elo": elo[name],
                    "wins": stats[name]["total_wins"],
                    "losses": stats[name]["total_losses"],
                    "draws": stats[name]["total_draws"],
                    "avg_ship_diff": stats[name]["avg_ship_diff"]
                }
                for i, name in enumerate(sorted(stats.keys(), key=lambda x: stats[x]["win_rate"], reverse=True))
            ],
            "matchups": {
                a: {b: self.results[a][b] for b in self.results[a]}
                for a in self.results
            }
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        print(f"  Results saved to: {filename}\n")


# ============================================================
# Bash Script Generator
# ============================================================

def generate_bash_script(agents, games=25, players=2, timeout=1.0, output_file="run_benchmark.sh"):
    """Generate a bash script to run benchmark tests."""
    
    agent_items = list(agents.items())
    
    # Generate all matchups (home & away)
    matchups = []
    for (name1, file1), (name2, file2) in itertools.combinations(agent_items, 2):
        matchups.append((name1, file1, name2, file2))
        matchups.append((name2, file2, name1, file1))
    
    script_lines = [
        "#!/bin/bash",
        "",
        "# Orbit Wars Benchmark Test Script",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Games per matchup: {games}",
        f"# Players: {players}",
        f"# Timeout: {timeout}s",
        "",
        "set -e  # Exit on error",
        "",
        "echo '========================================'",
        "echo '  Orbit Wars Benchmark Tests'",
        "echo '========================================'",
        "echo ''",
        f"echo 'Agents: {', '.join(agents.keys())}'",
        f"echo 'Games per matchup: {games}'",
        f"echo 'Players: {players}'",
        f"echo 'Timeout: {timeout}s'",
        "echo ''",
        "",
        "# Check if test_agent.py exists",
        "if [ ! -f \"test_agent.py\" ]; then",
        "    echo 'ERROR: test_agent.py not found!'",
        "    echo 'Please ensure test_agent.py is in the current directory.'",
        "    exit 1",
        "fi",
        "",
    ]
    
    # Add commands for each matchup
    if matchups:
        script_lines.append("# Run all matchups")
        script_lines.append("")
        
        for i, (name1, file1, name2, file2) in enumerate(matchups, 1):
            script_lines.append(f"echo '[{i}/{len(matchups)}] Testing {name1} vs {name2}...'")
            script_lines.append(
                f'python3 test_agent.py --agent {file1} --opponent {file2} '
                f'--games {games} --players {players} --timeout {timeout} --no-color'
            )
            script_lines.append("")
    else:
        # Only one agent, run self-test or skip
        script_lines.append("# Only one agent configured, skipping matchups")
        script_lines.append("echo 'Only one agent configured. Add more agents to AGENTS in benchmark.py to run matchups.'")
        script_lines.append("")
    
    script_lines.extend([
        "echo ''",
        "echo '========================================'",
        "echo '  All tests completed!'",
        "echo '========================================'",
    ])
    
    script_content = '\n'.join(script_lines) + '\n'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(output_file, 0o755)
    
    print(f"Bash script generated: {output_file}")
    print(f"Run it with: ./{output_file}")
    print("")
    print("Script contents:")
    print("-" * 60)
    print(script_content)
    print("-" * 60)


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Orbit Wars Round-Robin Tournament Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help=f"Games per matchup (default: {DEFAULT_GAMES})")
    parser.add_argument("--players", type=int, default=DEFAULT_PLAYERS, choices=[2, 4],
                        help=f"Players per game (default: {DEFAULT_PLAYERS})")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"Action timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--parallel", action="store_true",
                        help="Run matches in parallel (faster)")
    parser.add_argument("--include-random", action="store_true",
                        help="Include random agent in tournament")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--agents", type=str, nargs="+", default=None,
                        help="Specific agents to include (by name)")
    parser.add_argument("--generate-bash", action="store_true",
                        help="Generate bash script to run tests")
    parser.add_argument("--bash-output", type=str, default="run_benchmark.sh",
                        help="Output filename for generated bash script (default: run_benchmark.sh)")
    
    args = parser.parse_args()
    
    # Filter agents if specified
    agents = AGENTS
    if args.agents:
        agents = {k: v for k, v in AGENTS.items() if k in args.agents}
    
    # Check agent files exist
    missing = []
    for name, path in agents.items():
        if path != "random" and not os.path.isfile(path):
            missing.append(f"{name} ({path})")
    
    if missing:
        print(f"!  WARNING: Missing agent files: {', '.join(missing)}")
        print("   Please create these files or remove them from AGENTS config.\n")
    
    # Generate bash script if requested
    if args.generate_bash:
        generate_bash_script(
            agents=agents,
            games=args.games,
            players=args.players,
            timeout=args.timeout,
            output_file=args.bash_output
        )
        return
    
    # Run tournament
    runner = TournamentRunner(
        agents=agents,
        games=args.games,
        players=args.players,
        timeout=args.timeout,
        include_random=args.include_random,
        parallel=args.parallel
    )
    
    runner.run()
    
    if args.output:
        runner.save_results(args.output)


if __name__ == "__main__":
    main()
