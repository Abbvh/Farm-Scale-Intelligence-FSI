# ============================================
# FSI v1.0 — Farm Simulator
# Compressed time mode:
# Each cycle = 1 milking session (8-12 hours)
# Disease develops over 3-5 days (6-10 cycles)
# ============================================

import numpy as np
import requests
import time
import json
import random
from datetime import datetime

FSI_URL = "http://127.0.0.1:5000"

np.random.seed(42)

# Time compression settings
CYCLE_INTERVAL_SECONDS = 5      # Real seconds per cycle
HOURS_PER_CYCLE        = 8      # Farm hours represented per cycle
MILKING_SESSION        = True   # Each cycle = one milking session

class Cow:
    def __init__(self, cow_id, parity, breed, days_in_milk):
        self.cow_id       = cow_id
        self.parity       = parity
        self.breed        = breed
        self.days_in_milk = days_in_milk

        # Personal baseline
        self.baseline = {
            'milk_yield_kg':           np.random.normal(28, 3),
            'electrical_conductivity': np.random.normal(4.8, 0.2),
            'activity_steps':          int(np.random.normal(3800, 300)),
            'rumination_time':         int(np.random.normal(480, 30)),
            'body_temp':               np.random.normal(38.7, 0.15),
            'milk_fat_pct':            np.random.normal(3.9, 0.2),
            'milk_protein_pct':        np.random.normal(3.2, 0.1),
        }

        # Health state
        self.health_state      = 'healthy'
        self.disease_progress  = 0.0
        self.sessions_sick     = 0      # milking sessions sick
        self.last_prediction   = None
        self.alert_fired       = False

        # Disease onset timing — realistic 3-5 day progression
        # At 3 sessions/day that's 9-15 cycles
        self.progression_rate  = random.uniform(0.06, 0.12)

    def farm_time_label(self, cycle):
        """Convert cycle number to farm time"""
        total_hours = cycle * HOURS_PER_CYCLE
        days  = total_hours // 24
        hours = total_hours % 24
        return f"Day {days+1} — {hours:02d}:00"

    def get_readings(self):
        noise = lambda std: np.random.normal(0, std)

        if self.health_state == 'healthy':
            return {
                'parity':                  self.parity,
                'days_in_milk':            self.days_in_milk,
                'breed':                   self.breed,
                'milk_yield_kg':           max(5, self.baseline['milk_yield_kg'] + noise(1.5)),
                'electrical_conductivity': max(3.5, self.baseline['electrical_conductivity'] + noise(0.15)),
                'activity_steps':          max(500, int(self.baseline['activity_steps'] + noise(200))),
                'rumination_time':         max(200, int(self.baseline['rumination_time'] + noise(25))),
                'body_temp':               max(37.8, self.baseline['body_temp'] + noise(0.1)),
                'milk_fat_pct':            max(2.0, self.baseline['milk_fat_pct'] + noise(0.1)),
                'milk_protein_pct':        max(2.0, self.baseline['milk_protein_pct'] + noise(0.08)),
            }
        else:
            p = min(self.disease_progress, 1.0)
            return {
                'parity':                  self.parity,
                'days_in_milk':            self.days_in_milk,
                'breed':                   self.breed,
                'milk_yield_kg':           max(3, self.baseline['milk_yield_kg'] * (1 - 0.4*p) + noise(1.0)),
                'electrical_conductivity': min(9.0, self.baseline['electrical_conductivity'] + 2.5*p + noise(0.2)),
                'activity_steps':          max(300, int(self.baseline['activity_steps'] * (1 - 0.5*p) + noise(150))),
                'rumination_time':         max(150, int(self.baseline['rumination_time'] * (1 - 0.4*p) + noise(20))),
                'body_temp':               min(41.5, self.baseline['body_temp'] + 1.8*p + noise(0.15)),
                'milk_fat_pct':            max(1.5, self.baseline['milk_fat_pct'] * (1 - 0.2*p) + noise(0.1)),
                'milk_protein_pct':        min(5.0, self.baseline['milk_protein_pct'] * (1 + 0.15*p) + noise(0.08)),
            }

    def update_health(self):
        if self.health_state == 'healthy':
            # Realistic mastitis incidence ~25-40% per lactation
            # Per milking session: ~0.7% chance of onset
            if random.random() < 0.007:
                self.health_state     = 'deteriorating'
                self.disease_progress = 0.08
                print(f"⚠️  {self.cow_id} — subclinical mastitis onset (milking session)")

        elif self.health_state == 'deteriorating':
            # Progresses over 3-5 days (9-15 milking sessions)
            self.disease_progress += self.progression_rate
            self.sessions_sick    += 1

            if self.disease_progress >= 1.0:
                self.health_state = 'sick'
                print(f"🚨 {self.cow_id} — clinical mastitis (day {self.sessions_sick//3 + 1})")

            # Early natural recovery possible
            if self.disease_progress < 0.3 and random.random() < 0.08:
                self.health_state     = 'healthy'
                self.disease_progress = 0
                self.sessions_sick    = 0
                print(f"✅ {self.cow_id} — subclinical resolved naturally")

        elif self.health_state == 'sick':
            self.sessions_sick += 1
            # Treatment after ~1 day (3 milking sessions)
            if self.sessions_sick >= 3:
                self.health_state     = 'recovering'
                self.disease_progress = 0.6
                print(f"💊 {self.cow_id} — treatment initiated")

        elif self.health_state == 'recovering':
            self.disease_progress -= random.uniform(0.08, 0.15)
            if self.disease_progress <= 0:
                self.health_state     = 'healthy'
                self.disease_progress = 0
                self.sessions_sick    = 0
                self.alert_fired      = False
                print(f"✅ {self.cow_id} — fully recovered")


# Create herd of 20 cows
herd = []
for i in range(1, 21):
    cow_id = f"COW-{str(i).zfill(3)}"
    parity = random.randint(1, 5)
    breed  = random.choice([0, 1])
    dim    = random.randint(10, 300)
    herd.append(Cow(cow_id, parity, breed, dim))

print(f"🐄 FSI Farm Simulator — {len(herd)} cows")
print(f"⏱  Time compression: 1 cycle = {HOURS_PER_CYCLE}h farm time")
print(f"📡 Endpoint: {FSI_URL}")
print("=" * 50)

simulation_state = {
    'cows':        {},
    'alerts':      [],
    'last_update': None,
    'cycle':       0,
    'farm_time':   'Day 1 — 00:00',
    'hours_elapsed': 0,
    'compression_note': f'Demo mode — each cycle = {HOURS_PER_CYCLE}h farm time (1 milking session). Real deployment: 2-3x daily.'
}

def run_cycle():
    simulation_state['cycle']         += 1
    simulation_state['hours_elapsed'] += HOURS_PER_CYCLE
    simulation_state['last_update']    = datetime.now().strftime("%H:%M:%S")

    total_hours = simulation_state['hours_elapsed']
    days        = total_hours // 24
    hours       = total_hours % 24
    session_num = total_hours // 8
    session_label = ['Morning (06:00)', 'Afternoon (14:00)', 'Evening (22:00)'][session_num % 3]
    simulation_state['farm_time'] = f"Day {days+1} — {session_label}"

    for cow in herd:
        cow.update_health()
        readings = cow.get_readings()

        try:
            response = requests.post(
                f"{FSI_URL}/predict_realtime",
                json=readings,
                timeout=5
            )
            result = response.json()
            cow.last_prediction = result

            simulation_state['cows'][cow.cow_id] = {
                'cow_id':       cow.cow_id,
                'health_state': cow.health_state,
                'probability':  result.get('probability_pct', 0),
                'risk_level':   result.get('risk_level', 'low'),
                'readings':     readings,
                'shap_values':  result.get('shap_values', {}),
                'sessions_sick': cow.sessions_sick,
            }

            if result.get('risk_level') == 'high' and not cow.alert_fired:
                cow.alert_fired = True
                alert = {
                    'cow_id':      cow.cow_id,
                    'time':        datetime.now().strftime("%H:%M:%S"),
                    'farm_time':   simulation_state['farm_time'],
                    'probability': result.get('probability_pct', 0),
                    'message':     f"High mastitis risk — {result.get('probability_pct', 0)}% — {simulation_state['farm_time']}"
                }
                simulation_state['alerts'].insert(0, alert)
                simulation_state['alerts'] = simulation_state['alerts'][:10]
                print(f"🚨 ALERT: {cow.cow_id} — {result.get('probability_pct')}% — {simulation_state['farm_time']}")

        except Exception as e:
            print(f"Error {cow.cow_id}: {e}")

    print(f"Session {simulation_state['cycle']} — {simulation_state['farm_time']}")

if __name__ == '__main__':
    while True:
        run_cycle()
        time.sleep(CYCLE_INTERVAL_SECONDS)
