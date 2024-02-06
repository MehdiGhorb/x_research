from __future__ import division
import time
import Adafruit_PCA9685

# Helper function to set servo pulse width
def set_servo_pulse(channel, pulse):
    pulse_length = 1000000    # 1,000,000 us per second
    pulse_length //= 60       # 60 Hz
    pulse_length //= 4096     # 12 bits of resolution
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)


# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()

# Configure min and max servo pulse lengths
servo_min = 150  # Min pulse length out of 4096
servo_max = 450  # Max pulse length out of 4096

# Set frequency to 60hz, good for servos.
pwm.set_pwm_freq(60)

print('Moving servo motors on channels 0 and 1, press Ctrl-C to quit...')

try:
    while True:
        counter = 0
        # Execute servo movements based on the selected 'case'
        while counter < 15:
            # Move both servos between extremes simultaneously.
            for channel in range(0, 2):
                pwm.set_pwm(channel, 0, servo_min)
                
            time.sleep(1)  # Wait for 1 second
                
            for channel in range(0, 2):
                pwm.set_pwm(channel, 0, servo_max)
                
            time.sleep(1)  # Wait for 1 second
            counter += 1

        while 15 <= counter < 30:
            # Move the first servo to the maximum position.
            pwm.set_pwm(0, 0, servo_max)
            time.sleep(1)  # Wait for 1 second

            # Move the first servo back to the minimum position.
            pwm.set_pwm(0, 0, servo_min)
            time.sleep(1)  # Wait for 3 seconds

            counter += 1

        while 30 <= counter < 45:
            # Move the first servo to the maximum position.
            pwm.set_pwm(1, 0, servo_max)
            time.sleep(1)  # Wait for 1 second

            # Move the first servo back to the minimum position.
            pwm.set_pwm(1, 0, servo_min)
            time.sleep(1)  # Wait for 3 seconds

            counter += 1

except KeyboardInterrupt:
    print("Stopping servo movement.")

    # Set the servos to a neutral position before exiting.
    for channel in range(0, 2):
        pwm.set_pwm(channel, 0, (servo_min + servo_max) // 2)

