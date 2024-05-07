import random
import inflect
import sqlite3
import json

plural_singular = inflect.engine()

adding = []
subtracting = []
updating = []
deleting = []
failuers = []

opener = ['Sure, ', 'Alright, ', 'Cerntainly, ', 'Of course,', 'No problem, ']
second_opener = ['executing the following actions:\n',
                 'making those changes:\n', 'updating the list:\n']


def convert_units(old_unit, new_unit, new_amount):  # old, new, new
    """
    Convert the amount new_amount from new_unit to old_unit.

    Args:
    old_unit (str): The unit to convert to (e.g., 'g', 'kg', 'L', 'mL').
    q1 (float): The amount in old_unit (only used to ensure compatibility of units).
    new_unit (str): The unit to convert from (e.g., 'g', 'kg', 'L', 'mL').
    new_amount (float): The amount in new_unit to be converted.

    Returns:
    float: The amount of new_amount converted to old_unit.
    """
    # Dictionary to store conversion factors to a base unit (gram for mass, liter for volume)
    conversions = {
        'g': 1,       # gram is the base unit for mass
        'Kg': 1000,   # 1 kg = 1000 grams
        'kg': 1000,
        'Liter': 1,  # liter is the base unit for volume
        'Litre': 1,
        'L': 1,
        'mL': 0.001   # 1 mL = 0.001 liters
    }

    if old_unit == 'unit':
        return new_unit, new_amount
    elif new_unit == 'unit':
        return old_unit, new_amount
    # Check if both units are of the same category (both mass or both volume)
    elif ((old_unit in ['g', 'kg', 'Kg'] and new_unit in ['g', 'kg', 'Kg']) or
            (old_unit in ['L', 'mL', 'Liter'] and new_unit in ['L', 'mL', 'Liter'])):
        # Convert new_amount to the base unit
        base_amount = new_amount * conversions[new_unit]
        # Convert from base unit to old_unit
        converted_amount = base_amount / conversions[old_unit]
        return old_unit, converted_amount
    else:
        return new_unit, new_amount
        # raise ValueError(
        #     "Incompatible units. Please make sure both units are either both mass or both volume.")


def add_item(grocery, amount, unit, retrieved_list, chat_id):
    if amount == 0:
        amount = 1
    elif amount < 0:
        sub_item(grocery, amount, unit)
        return
    for dictionary in retrieved_list:
        # add amount to existing grocery
        if grocery in dictionary:
            print(dictionary[grocery])
            # convert units if needed
            if dictionary[grocery][1] != unit:
                unit, amount = convert_units(
                    dictionary[grocery][1], unit, amount)
            dictionary[grocery] = (round(
                dictionary[grocery][0] + amount, 2), unit)

            # string to output list
            if amount > 1 and dictionary[grocery][1] == 'unit':
                adding.append(
                    f'{amount} more {plural_singular.plural(grocery)}')
            elif amount <= 1 and dictionary[grocery][1] == 'unit':
                adding.append(f'{amount} more {grocery}')
            elif amount <= 1:
                adding.append(
                    f'{amount} more {dictionary[grocery][1]} of {plural_singular.plural(grocery)}')
            else:
                adding.append(
                    f'{amount} more {dictionary[grocery][1]}s of {plural_singular.plural(grocery)}')
            update_db(retrieved_list, chat_id)
            return True

    # add new grocery
    prodct_dict = {grocery: (round(amount, 2), unit)}
    retrieved_list.append(prodct_dict)

    # string to output list
    if amount > 1 and unit == 'unit':
        adding.append(f'{amount} {plural_singular.plural(grocery)}')
    elif amount <= 1 and unit == 'unit':
        adding.append(f'{amount} {grocery}')
    elif amount <= 1:
        adding.append(
            f'{amount} {unit} of {plural_singular.plural(grocery)}')
    else:
        adding.append(
            f'{amount} {unit} of {plural_singular.plural(grocery)}')

    update_db(retrieved_list, chat_id)
    return True


def sub_item(grocery, amount, unit, retrieved_list, chat_id):
    # amount
    if amount < 0:
        amount = -amount
    elif amount == 0 or amount == 'null':
        amount = 1
    for dictionary in retrieved_list:
        if grocery in dictionary:
            # units
            if dictionary[grocery][1] != unit:
                unit, amount = convert_units(
                    dictionary[grocery][1], unit, amount)

            # subtraction
            dictionary[grocery] = (round(
                dictionary[grocery][0] - amount, 2), dictionary[grocery][1])

            # Negative or zero -> delete
            if dictionary[grocery][0] <= 0:
                check = delete_item(grocery, retrieved_list, chat_id)
                return check
            else:
                if amount <= 1:
                    if unit == 'unit':
                        subtracting.append(f'{amount} {grocery}')
                    else:
                        subtracting.append(
                            f'{amount} {dictionary[grocery][1]} of {plural_singular.plural(grocery)}')
                else:  # amount > 1
                    if unit == 'unit':
                        subtracting.append(
                            f'{amount} {plural_singular.plural(grocery)}')
                    else:
                        subtracting.append(
                            f'{amount} {dictionary[grocery][1]} of {plural_singular.plural(grocery)}')

            update_db(retrieved_list, chat_id)
            return True


def update_item(grocery, amount, unit, retrieved_list, chat_id):
    # amount == 0
    if amount == 0:
        check = delete_item(grocery, retrieved_list, chat_id)
        return check
    else:
        for dictionary in retrieved_list:
            if grocery in dictionary:
                dictionary[grocery] = (round(amount, 2), unit)
                if amount <= 1:
                    updating.append(
                        f'{plural_singular.plural(grocery)} to {amount} {unit}')
                # amount > 1
                else:
                    updating.append(
                        f'{plural_singular.plural(grocery)} to {amount} {unit}')
                update_db(retrieved_list, chat_id)
                return True

        check = add_item(grocery, amount, unit, retrieved_list, chat_id)
        return check


def delete_item(grocery, retrieved_list, chat_id):
    print(grocery)
    for dictionary in retrieved_list:
        print(dictionary)
        if grocery in dictionary:
            retrieved_list.remove(dictionary)
            deleting.append(grocery)

            update_db(retrieved_list, chat_id)
            return True
    failuers.append(
        f'It seems like there are no {plural_singular.plural(grocery)} in your list.\n')
    return False


def update_db(retrieved_list, chat_id):
    conn = sqlite3.connect('grocerybot.db')
    c = conn.cursor()
    json_data = json.dumps(retrieved_list)
    c.execute(
        "INSERT OR REPLACE INTO groceries (chat_id, list_data) VALUES (?, ?)", (chat_id, json_data))
    conn.commit()
    conn.close()


def is_list_empty(l):
    """
    Function to check if a dictionary is empty.
    """
    return not bool(l)


def answer_to_list(response, retrieved_list, chat_id):
    if isinstance(response, list):
        # Iterate over the dictionary and append each grocery to the list
        if is_list_empty(response):
            print("I'm sorry, i couldn't find any groceries.")
            answer = "I'm sorry. I couldn't find groceries in the sentence.\n"
        else:
            # Reset lists
            adding.clear()
            subtracting.clear()
            updating.clear()
            deleting.clear()
            failuers.clear()

            # counters
            action_counter = len(response)
            success = 0

            # loop each grocery in response
            for prod in response:
                action = False
                print(prod)

                # default amount = 1
                if prod['amount'] == 'None' or prod['amount'] == 0:
                    prod['amount'] = 1
                # default action = 'add'
                if 'action' not in prod:
                    prod['action'] = 'add'
                # default units = 'unit'
                if prod['unit'] == '' or prod['unit'] == 'unknown' or prod['unit'] == None:
                    prod['unit'] = 'unit'

                # Call function by action
                if prod['action'].lower() == 'add':
                    action = add_item(
                        prod['grocery'].lower(), prod['amount'], prod['unit'].lower(), retrieved_list, chat_id)
                elif prod['action'].lower() == 'update':
                    action = update_item(
                        prod['grocery'].lower(), prod['amount'], prod['unit'].lower(), retrieved_list, chat_id)
                elif prod['action'].lower() == 'subtract':
                    action = sub_item(
                        prod['grocery'].lower(), prod['amount'], prod['unit'].lower(), retrieved_list, chat_id)
                elif prod['action'].lower() == 'delete':
                    action = delete_item(
                        prod['grocery'].lower(), retrieved_list, chat_id)

                # How many groceries succeed
                if action:
                    success += 1

            # OUTPUT
            ans = output_reponse(action_counter, success)
            return ans


def output_reponse(action_counter, success):
    # Opener if some or all actions completed
    if success >= 1:
        rand1 = random.randint(0, len(opener)-1)
        # rand2 = random.randint(0, 2)

        # second_opener[rand2] + '\n'
        answer = opener[rand1] + '\n'

        # add, subtract, update, delete
        # Adding
        if len(adding) >= 1:
            # 1 - OPEN
            answer += 'adding '
            # 2 - LOOP
            for i in range(len(adding)):
                # Item (amount, unit, grocery)
                answer += adding[i]
                # spacing
                if i < len(adding)-2:
                    answer += ', '
                elif i < len(adding)-1:
                    answer += ' and '
            # 3 - END
            answer += ' to the list.\n'

            if (len(subtracting) > 0 and len(updating) == 0 and len(deleting) == 0) or (len(subtracting) == 0 and len(updating) > 0 and len(deleting) == 0) or (len(subtracting) == 0 and len(updating) == 0 and len(deleting) > 0):
                answer += 'And '

        # Subtracting
        if len(subtracting) >= 1:
            # 1 - OPEN
            answer += 'subtracting '
            # 2 - LOOP
            for i in range(len(subtracting)):
                # Item (amount, unit, grocery)
                answer += subtracting[i]
                # spacing
                if i < len(subtracting)-2:
                    answer += ', '
                elif i < len(subtracting)-1:
                    answer += ' and '
            # 3 - END
            answer += ' from the list.\n'

            if (len(updating) > 0 and len(deleting) == 0) or (len(updating) == 0 and len(deleting) > 0):
                answer += 'And '

        # Updating
        if len(updating) >= 1:
            # 1 - OPEN
            answer += 'changing '
            # 2 - LOOP
            for i in range(len(updating)):
                # Item (grocery, amount, unit)
                answer += updating[i]
                # spacing
                if i < len(updating)-2:
                    answer += ', '
                elif i < len(updating)-1:
                    answer += ' and '
            # 3 - END
            answer += '.\n'

            if (len(deleting) > 0):
                answer += 'And '

        # Delete
        if len(deleting) >= 1:
            # 1 - OPEN
            answer += 'deleting '
            # 2 - LOOP
            for i in range(len(deleting)):
                # Item
                answer += deleting[i]
                # spacing
                if i < len(deleting)-2:
                    answer += ', '
                elif i < len(deleting)-1:
                    answer += ' and '
            # 3 - END
            answer += ' from the list.\n'

    # Errors for when some actions were completed and some weren't
    if len(failuers) >= 1:
        answer += '\nAlso, please notice the incomplete actions:'
        for j in range(len(failuers)):
            answer += '\n-'
            answer += failuers[j]

    # Errors for when no actions were completed
    if success == 0:
        answer = "I'm sorry. I couldn't complete the following actions:\n"
        for j in range(len(failuers)):
            answer += f'\n-'
            answer += failuers[j]

    return answer


def print_list(groceries_list):
    lines = []
    text_str = ''
    for grocery in groceries_list:
        for key, value in grocery.items():
            amount, unit = value
            # Adding 's' to the unit if the amount is more than 1
            if amount > 1 and unit.endswith('unit'):
                unit += 's'
            lines.append(f"{key}: {amount} {unit}")

    text_str = "\n".join(lines)
    print(text_str)
    return text_str
