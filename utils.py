import numpy as np
from datetime import date

def date_norm(iso):
	year, month, day = tuple(iso.split('-'))
	time = date(int(year), int(month), int(day))

	min_date = date(2015, 1, 3)
	max_date = date(2018, 3, 25)
	max_delta = (max_date - min_date).days

	delta = (time - min_date).days

	return delta / max_delta

def lower_upper_bound(array):
	q1 = np.percentile(array, 25)
	q3 = np.percentile(array, 75)
	lower_bound = 2.5 * q1 - 1.5 * q3
	upper_bound = 2.5 * q3 - 1.5 * q1
	return lower_bound, upper_bound

def min_max(array):
	return min(array), max(array)
