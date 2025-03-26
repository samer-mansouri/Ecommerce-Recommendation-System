from app.engine import RecommendationEngine
from datetime import datetime
import time

RETRAIN_INTERVAL_HOURS = 2

def main():
    engine = RecommendationEngine()

    while True:
        print(f"[{datetime.now().isoformat()}] Starting model retrain...")
        try:
            engine.train("retrain")
            print(f"[{datetime.now().isoformat()}] Model retrain completed.")
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] Retraining failed: {e}")

        # Sleep for the given interval
        sleep_time = RETRAIN_INTERVAL_HOURS * 3600
        print(f"Sleeping for {sleep_time} seconds...\n")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
