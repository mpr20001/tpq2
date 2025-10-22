#!/bin/bash
eval `tail -n +2 /meta/aws-iam/credentials | head -n -1 | sed -r 's/^(.*)\ \=\ /\U\1\E=/' | sed -r 's/^/export /'`