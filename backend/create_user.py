"""Create or update a user (asks for the password):
    python -m backend.create_user admin --role admin
    python -m backend.create_user alice
    python -m backend.create_user alice --update      # change password / role
"""
import argparse
import getpass

from . import auth

ap = argparse.ArgumentParser()
ap.add_argument("username")
ap.add_argument("--role", choices=["admin", "user"], default="user")
ap.add_argument("--update", action="store_true", help="change an existing user")
args = ap.parse_args()

password = getpass.getpass("Password (min 8 characters): ")
if password != getpass.getpass("Repeat password: "):
    raise SystemExit("Passwords do not match")
try:
    auth.create_user(args.username, password, args.role, args.update)
except ValueError as e:
    raise SystemExit(str(e))
print(f"Saved {args.role} '{args.username.strip().lower()}'")
