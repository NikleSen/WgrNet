from datetime import datetime

def get_date_time_str():
    date_time = datetime.now()
    str_year = str(date_time.date().year).zfill(4)
    str_month = str(date_time.date().month).zfill(2)
    str_day = str(date_time.date().day).zfill(2)
    str_date = str_year + '-' + str_month + '-' + str_day

    str_time = str(str(date_time.time().hour).zfill(2) +
                   '-' + str(date_time.time().minute).zfill(2) + '-' + str(date_time.time().second).zfill(2))
    return str_date + '-' + str_time


def get_h_m_s(end, start):
    h, remainder = divmod((end - start).seconds, 3600)
    m, s = divmod(remainder, 60)
    h = str(h).zfill(2)
    m = str(m).zfill(2)
    s = str(s).zfill(2)
    return h, m, s


if __name__ == '__main__':
    print(get_date_time_str())
    end = datetime.now()
    print(end)