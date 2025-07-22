#!/bin/bash
# This script sets up the v4l2loopback device.

# Exit immediately if a command exits with a non-zero status.
set -e

# Device number and card label can be customized
DEVICE_NUMBER=10
CARD_LABEL="VirtualCam"

echo "Checking for v4l2loopback module..."

if lsmod | grep -q "v4l2loopback"; then
    echo "v4l2loopback module is already loaded."
else
    echo "Loading v4l2loopback module..."
    sudo modprobe v4l2loopback
fi

echo "Creating virtual webcam device /dev/video${DEVICE_NUMBER}..."

# Unload the module if it's already creating a device with this number
# This is a simple way to ensure we can set our desired parameters
if [ -e "/dev/video${DEVICE_NUMBER}" ]; then
    echo "Device /dev/video${DEVICE_NUMBER} already exists. It might be in use."
    echo "You may need to unload the module manually if you want to change parameters:"
    echo "sudo modprobe -r v4l2loopback"
else
    # Create the virtual device
    sudo modprobe v4l2loopback devices=1 video_nr=${DEVICE_NUMBER} card_label="${CARD_LABEL}" exclusive_caps=1
    echo "Virtual webcam created at /dev/video${DEVICE_NUMBER} with label '${CARD_LABEL}'."
fi

echo "To verify, you can run:"
echo "v4l2-ctl --list-devices"
echo "Or check with a webcam application." 