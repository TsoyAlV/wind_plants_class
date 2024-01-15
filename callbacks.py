import subprocess

def callback_congratulations_message_to_telegram():
    subprocess.run(["telegram-send", "'Выполнилось обучение, проверь расчет!'"])
