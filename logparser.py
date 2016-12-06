import matplotlib.pyplot as plt

def filter(lines):
    filtered = []
    for l in lines:
        parts = l.split(',')
        if float(parts[-1].strip().replace(')','')) < trth and float(parts[-2].strip().replace(')', '')) < teth:
            filtered.append(l)
    return filtered

def analyze(lines):
    x = [float(l.split(',')[-1].strip().replace(')', '')) for l in lines]
    y = [float(l.split(',')[-2].strip().replace(')', '')) for l in lines]
    print len(x)
    print len(y)

    plt.plot(range(len(x[300:400])), x[300:400], 'b-')
    plt.plot(range(len(y[300:400])), y[300:400], 'g-')
    plt.show()

    #for line in get_min_test_error(lines)[:30]:
    #   print line


def get_min_test_error(lines):
    return sorted(lines,
                  key=lambda x: float(x.split(',')[-1].strip().replace(')', '')))

    # get min test error sorted that the -5th field (depth) greater than 30 doesn't show up on top
    #lambda x: 1000.0 if int(x.split(',')[-5].strip()) >= 30 else float(x.split(',')[-1].strip().replace(')', '')))


def main():
    f = open('dtree-logs/low_errs', 'r')
    lines = f.readlines()
    f.close()

    analyze(lines)


if __name__ == '__main__':
    main()
