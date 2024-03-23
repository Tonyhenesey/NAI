def load_data(filepath):
    with open(filepath, 'r') as file:
        data = [line.strip().split(',') for line in file if line.strip()]
        for row in data:
            row[:4] = [float(num) for num in row[:4]]
    return data

def euclidean_distance(v1, v2):
    return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5

def kNN(train_data, test_instance, k):
    distances = []
    for train_instance in train_data:
        dist = euclidean_distance(test_instance[:-1], train_instance[:-1])
        distances.append((train_instance, dist))
    distances.sort(key=lambda x: x[1])

    neighbors = [distances[i][0] for i in range(k)]
    votes = {}
    for neighbor in neighbors:
        response = neighbor[-1]
        if response in votes:
            votes[response] += 1
        else:
            votes[response] = 1
    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]

def calculate_accuracy(train_data, test_data, k):
    correct = 0
    for test_instance in test_data:
        prediction = kNN(train_data, test_instance, k)
        if prediction == test_instance[-1]:
            correct += 1
    accuracy = (correct / len(test_data)) * 100
    return accuracy

def manual_input(train_data):
    k = int(input("Wprowadź wartość k dla algorytmu kNN: "))
    test_instance = input("Wprowadź wektor testowy (np. 5.1,3.5,1.4,0.2): ")
    test_instance = [float(num) for num in test_instance.split(',')] + ['?']
    prediction = kNN(train_data, test_instance, k)
    print(f"Przewidziana klasa: {prediction}")

def file_input(train_data):
    k = int(input("Podaj wartość k dla algorytmu kNN: "))
    filepath = input("Podaj ścieżkę do pliku z wektorami testowymi: ")
    test_data = load_data(filepath)
    accuracy = calculate_accuracy(train_data, test_data, k)
    print(f"Celność modelu: {accuracy}%")

def main():
    train_data_path = 'iris-train.txt'
    train_data = load_data(train_data_path)

    while True:
        choice = input("Wybierz opcję: [1] Wprowadź wektor testowy, [2] Wczytaj wektory z pliku, [3] Zakończ: ")
        if choice == "1":
            manual_input(train_data)
        elif choice == "2":
            file_input(train_data)
        elif choice == "3":
            print("Zakończenie programu.")
            break
        else:
            print("Nieprawidłowa opcja, spróbuj ponownie.")

if __name__ == "__main__":
    main()
