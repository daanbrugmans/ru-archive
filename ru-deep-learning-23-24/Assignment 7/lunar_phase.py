"""
lunar_phase.py - Calculate Lunar Phase
Author: Sean B. Palmer, inamidst.com, edited by Daan Brugmans
Cf. http://en.wikipedia.org/wiki/Lunar_phase#Lunar_phase_calculation
"""

import math, decimal, datetime
dec = decimal.Decimal

def position(now=None): 
    if now is None: 
        now = datetime.datetime.now()

    diff = now - datetime.datetime(2001, 1, 1)
    days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
    lunations = dec("0.20439731") + (days * dec("0.03386319269"))

    return lunations % dec(1)

def get_lunar_phase_name() -> str: 
    pos = position()
    
    index = (pos * dec(8)) + dec("0.5")
    index = math.floor(index)
   
    return {
      0: "new_moon", 
      1: "waxing_crescent", 
      2: "first_quarter", 
      3: "waxing_gibbous", 
      4: "full_moon", 
      5: "waning_gibbous", 
      6: "last_quarter", 
      7: "waning_crescent"
    }[int(index) & 7]

   