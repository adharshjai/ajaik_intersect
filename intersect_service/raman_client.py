"""
Example INTERSECT client: send spectrum CSV exports to the Raman classifier and print the results.

Run from the repository root (with the service already running):
    python -m intersect_service.raman_client data/example/spectra/ethanol_S1_01.csv [more.csv ...]
"""
import argparse
import json
import logging
import os

from intersect_sdk import (
    INTERSECT_RESPONSE_VALUE,
    IntersectClient,
    IntersectClientCallback,
    IntersectClientConfig,
    IntersectDirectMessageParams,
    default_intersect_lifecycle_loop,
)

from raman.data import read_spectrum_csv

logging.basicConfig(level=logging.INFO)


def parse_args():
    p = argparse.ArgumentParser(description='Send Raman spectra to the classifier service')
    p.add_argument('csv_files', nargs='+')
    p.add_argument('--destination', default='ajaik.lab.raman.analysis.substance-classifier',
                   help='<organization>.<facility>.<system>.<subsystem>.<service> of the classifier')
    p.add_argument('--broker-host', default=os.environ.get('INTERSECT_BROKER_HOST', '127.0.0.1'))
    p.add_argument('--broker-port', type=int, default=int(os.environ.get('INTERSECT_BROKER_PORT', 1883)))
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    messages = []
    for path in args.csv_files:
        shift, intensity = read_spectrum_csv(path)
        messages.append(IntersectDirectMessageParams(
            destination=args.destination,
            operation='RamanClassifier.classify_spectrum',
            payload={'raman_shift': shift.tolist(), 'intensity': intensity.tolist(),
                     'spectrum_id': os.path.basename(path)},
        ))

    remaining = len(messages)

    def on_response(_source: str, _operation: str, has_error: bool, payload: INTERSECT_RESPONSE_VALUE) -> None:
        global remaining
        print(f'ERROR: {payload}' if has_error else json.dumps(payload, indent=2))
        remaining -= 1
        if remaining == 0:
            raise Exception  # all answers received: break out of the message loop (SDK convention)

    config = IntersectClientConfig(
        initial_message_event_config=IntersectClientCallback(messages_to_send=messages),
        brokers=[{
            'host': args.broker_host,
            'port': args.broker_port,
            'username': os.environ.get('INTERSECT_BROKER_USERNAME', 'intersect_username'),
            'password': os.environ.get('INTERSECT_BROKER_PASSWORD', 'intersect_password'),
            'protocol': 'mqtt5.0',
        }],
    )
    default_intersect_lifecycle_loop(IntersectClient(config=config, user_callback=on_response))
