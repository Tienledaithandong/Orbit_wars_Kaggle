#!/bin/bash

# Orbit Wars Benchmark Test Script
# Generated: 2026-04-30 08:14:38
# Games per matchup: 25
# Players: 2
# Timeout: 1.0s

set -e  # Exit on error

echo '========================================'
echo '  Orbit Wars Benchmark Tests'
echo '========================================'
echo ''
echo 'Agents: ulti, random'
echo 'Games per matchup: 25'
echo 'Players: 2'
echo 'Timeout: 1.0s'
echo ''

# Check if test_agent.py exists
if [ ! -f "test_agent.py" ]; then
    echo 'ERROR: test_agent.py not found!'
    echo 'Please ensure test_agent.py is in the current directory.'
    exit 1
fi

# Run all matchups

echo '[1/2] Testing ulti vs random...'
python3 test_agent.py --agent main_ulti.py --opponent random --games 25 --players 2 --timeout 1.0 --no-color

echo '[2/2] Testing random vs ulti...'
python3 test_agent.py --agent random --opponent main_ulti.py --games 25 --players 2 --timeout 1.0 --no-color

echo ''
echo '========================================'
echo '  All tests completed!'
echo '========================================'
