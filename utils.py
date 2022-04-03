import re

def format_text(text):
    regex = re.compile(r'[\n\r\t]')
    text = regex.sub('', text)

    return " ".join(text.split())


def format_currency(value):
    value = value.replace('€', '')
    value = value.replace('?', '')
    value = value.replace('Loan fee:', '')
    value = value.replace('loan transfer', '')
    value = value.replace('free transfer', '')
    
    if not value:
        return 0
    
    if value[-1] == 'm':
        value = value.replace('m', '')
        return int(float(value) * 1000000)
    
    if value[-2:] == 'bn':
        value = value.replace('bn', '')
        return int(float(value) * 1000000000)

    if value[-1] == '.':
        value = value.replace('.', '')
        if value[-2:] == 'Th':
            value = value.replace('Th', '')
            return int(value) * 1000
    
    return int(value)
