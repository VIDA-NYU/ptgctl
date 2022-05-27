import sys
import logging


NC='\033[0m'
def color(text, color='grey', brightness=0):
    return f'{color_code(color, brightness)}{text}{NC}'

def color_code(color='grey', brightness=0):
    i = 1 if brightness > 0 else 2 if brightness < 0 else 0
    c = COLOR_CODES[color.lower()]
    return f'\033[{i};{c}m'

COLOR_CODES = {
    'red':    30,
    'orange': 31,
    'green':  32,
    'yellow': 33,
    'blue':   34,
    'purple': 35,
    'cyan':   36,
    'grey':   37,
}

class ColorFormatter(logging.Formatter):
    LEVELS = {
        logging.DEBUG: color_code('purple', 0), 
        logging.INFO: color_code('grey', -1), 
        logging.WARNING: color_code('orange', 1), 
        logging.ERROR: color_code('red', 0), 
        logging.CRITICAL: color_code('red', 1), 
    }
    def format(self, record):
        return self.color(super().format(record), record.levelno)

    def color(self, text, level):
        color = self.LEVELS.get(level)
        return f'{color}{text}{NC}' if color else text

def getLogger(name=__name__.split('.')[0], level='info'):
    '''Get a logger.
    '''
    log = logging.getLogger(name)
    log.propagate = False
    if log.handlers:
        return log
    log_handler = logging.StreamHandler(sys.stderr)
    formatter = ColorFormatter('%(message)s')
    log_handler.setFormatter(formatter)
    log.addHandler(log_handler)
    log.setLevel(aslevel(level))
    def log_color(text, level):
        return formatter.color(text, aslevel(level))
    log.color = log_color
    return log

def aslevel(level):
    return logging._nameToLevel.get(
        level.upper() if isinstance(level, str) and level not in logging._nameToLevel else level, 
        level)
