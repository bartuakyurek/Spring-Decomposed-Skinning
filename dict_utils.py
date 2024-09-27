#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:15:07 2024

@author: bartu
"""

def _count_key_root_occurence_in_dict(dictionary : dict,
                                       key_seperator  : str,
                                       root_name   : str ) -> int:
    assert type(root_name) is str, f"Expected str in root_name, got {type(root_name)}"
    
    n_instance = 0
    for key in dictionary:
        
        key_root = key.split(key_seperator)[0]
        
        if key_root == root_name:
            n_instance += 1
            
    return n_instance