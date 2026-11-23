"""CSC350 Lab 3: NumPy hands-on
Name:
Roll number:

Run this file after every task:  python lab3_numpy.py
Commit after every task (git add, git commit -m "Task N done").
"""
import numpy as np

print("NumPy version:", np.__version__)

# =====================================================================
# Task 1: Creating arrays (2 marks)
# =====================================================================
print("\n=== Task 1 ===")
# 1a. Make an array `temps` with this week's temperatures:
#     31, 28, 35, 30, 33, 29, 36
#     Print it, then print its dtype, shape and size.
temps =np.array([31, 28, 35, 30, 33, 29, 36])
print("temperature :", temps)
print("Data type :", temps.dtype)
print("shape of array :", temps.shape)
print("Size of array:", temps.size)

# 1b. Make `months`: the numbers 1 to 12 as a 3 x 4 table
#     (use np.arange and reshape). Print it and its shape.
months = np.arange(1, 13).reshape(3, 4)
print("Months array:\n", months)
print("Shape of months array:", months.shape)

# 1c. Print an array of five zeros, and a 2 x 3 array of ones.
print("Array of five zeros:", np.zeros(5))
print("2 x 3 array of ones:\n", np.ones((2, 3)))

# 1d. Make a random generator with seed 42 and roll 10 dice
#     (whole numbers from 1 to 6). Print the result.
np.random.seed(42)
dice_rolls = np.random.randint(1, 7, size=10)
print("Dice rolls:", dice_rolls)


# =====================================================================
# Task 2: Indexing and masks (2.5 marks)  (use `temps` from Task 1)
# =====================================================================
print("\n=== Task 2 ===")
# 2a. Print the first and the last temperature.
print("First temperature:", temps[0])
print("Last temperature:", temps[-1])
# 2b. Print the weekend temperatures (the last two days).
print("Weekend temperatures:", temps[-2:])
# 2c. Print the temperatures in reverse order.
print("Temperatures in reverse order:", temps[::-1])
# 2d. Print only the hot days (above 32), how many there are,
#     and their average.  No loops!
hot_days = temps[temps > 32]
print("Hot days:", hot_days)
print("Number of hot days:", len(hot_days))
print("Average of hot days:", np.mean(hot_days))
# 2e. Print the temperatures from 29 to 33 (both included). Hint: & 
print("Temperatures from 29 to 33:", temps[(temps >= 29) & (temps <= 33)])
# 2f. Use np.where to label each day "Hot" (33 or more) or "OK".
labels = np.where(temps >= 33, "Hot", "OK")
print("Labels for each day:", labels)



# =====================================================================
# Task 3: Gradebook (3 marks)
# Rows = students, columns = 4 subjects.
# =====================================================================
print("\n=== Task 3 ===")
names = np.array(["Ali", "Sara", "Usman", "Hina", "Bilal"])
marks = np.array([[78, 85, 62, 90],
                  [92, 88, 95, 79],
                  [45, 60, 48, 70],
                  [88, 94, 91, 85],
                  [65, 42, 75, 58]])
# 3a. Print the shape of marks.
print("Shape of marks:", marks.shape)
# 3b. Print Usman's marks (one row) and all marks of subject 3
#     (column index 2).
print("Usman's marks:", marks[2])
print("Subject 3 marks:", marks[:, 2])
# 3c. Print the average of each subject and the average of each student.
print("Average of each subject:", np.mean(marks, axis=0))
print("Average of each student:", np.mean(marks, axis=1))
# 3d. Print the NAME of the student with the highest average.
#     Hint: argmax gives a position; use it on `names`.
highest_student = names[np.argmax(np.mean(marks, axis=1))]
print("Student with highest average:", highest_student)
# 3e. Subject weights are 0.3, 0.3, 0.2, 0.2. Print each student's
#     weighted total, rounded to 1 decimal. (Broadcasting!)
weights = np.array([0.3, 0.3, 0.2, 0.2])
weighted_totals = np.round(np.sum(marks * weights, axis=1), 1)
print("Weighted totals:", weighted_totals)
# 3f. Add 5 grace marks to everyone, but no mark may go above 100.
#     Print the new table. Hint: np.clip
marks += 5
marks = np.clip(marks, None, 100)
print("New marks table:\n", marks)
# 3g. Print how many students failed (below 50) in each subject.
failed_counts = np.sum(marks < 50, axis=0)
print("Number of students who failed each subject:", failed_counts)


# =====================================================================
# Task 4: An image is an array (2.5 marks)
# A 5 x 5 grayscale image: 0 = black, 255 = white.
# =====================================================================
print("\n=== Task 4 ===")
img = np.array([[ 10,  10, 200,  10,  10],
                [ 10,  10, 220,  10,  10],
                [210, 230, 250, 230, 210],
                [ 10,  10, 220,  10,  10],
                [ 10,  10, 200,  10,  10]])
# 4a. Print the shape, the darkest value and the brightest value.
print("Shape of image:", img.shape)
print("Darkest value:", np.min(img))
print("Brightest value:", np.max(img))
# 4b. Print the inverted image (white becomes black). Hint: 255 - ...
inverted_img = 255 - img
print("Inverted image:\n", inverted_img)
# 4c. Print a black-and-white version: 1 where the pixel is above 128,
#     else 0. Hint: a mask, then .astype(int)
bw_img = (img > 128).astype(int)
print("Black-and-white image:\n", bw_img)
# 4d. Print row 2 normalized to the range 0 to 1, rounded to 2 decimals.
row_2_normalized = np.round((img[2] - np.min(img)) / (np.max(img) - np.min(img)), 2)
print("Row 2 normalized:", row_2_normalized)
# 4e. Print the 3 x 3 centre of the image (crop).
centre_crop = img[1:4, 1:4]
print("3 x 3 centre of the image:\n", centre_crop)
