import tkinter as tk
import time

# Create the main window
window = tk.Tk()
window.title('Timer')

# Create a function to start the timer
def start_timer():
    global start_time
    start_time = time.time()
    update_time()

# Create a function to stop the timer
def stop_timer():
    global start_time
    start_time = None

# Create a function to reset the timer
def reset_timer():
    global elapsed_time
    elapsed_time = 0
    time_label.config(text='00:00:00')

# Create a function to update the time display
def update_time():
    global elapsed_time
    global start_time
    if start_time:
        elapsed_time = time.time() - start_time
        time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        time_label.config(text=time_string)
        window.after(1000, update_time)
        
# Create a label to display the time
time_label = tk.Label(window, text='00:00:00', font=('Helvetica', 32))
time_label.pack()

# Create a start button
start_button = tk.Button(window, text='Start', font=('Helvetica', 16), command=start_timer)
start_button.pack()

# Create a stop button
stop_button = tk.Button(window, text='Stop', font=('Helvetica', 16), command=stop_timer)
stop_button.pack()

# Create a reset button
reset_button = tk.Button(window, text='Reset', font=('Helvetica', 16), command=reset_timer)
reset_button.pack()

# Create variables to store the start time and elapsed time
start_time = None
elapsed_time = 0


# Run the main loop
window.mainloop()
