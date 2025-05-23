# Аппроксимация и оптимизация функции `logf`

## Окружение

Использовалась машина с Intel(R) Core(TM) i5-11400F @ 2.60GHz, поддерживающим в том числе расширения AVX2 и AVX512.
Операционная система: Arch Linux kernel 6.13.8. Компилятор: Clang 19.

## Сборка и тестирование
Гарантируется корректная сборка проекта на x64-машинах с
расширениями AVX2 и AVX512 с Linux с компилятором, указанным в окружении выше.

```sh
python3 -m venv .venv
pip install conan
conan profile detect --force
conan install . --build=missing --output-folder=build -pr:a=linux_release.profile
cmake . --preset conan-release
cmake --build build -j
ctest --test-dir=build --output-on-failure -j
```

## Тестирование точности

Тест точности можно найти [здесь](tests/src/logf_test.cc).
В качестве референса для проверки аппроксимации использовалась `boost::multiprecision` версии 1.87.0
в режиме двойной точности по IEEE754. Проверены все представимые числа типа `float` в диапазоне
$[2^{-126}, +\infty)$.

Результаты показали что удалось добиться точности 1.0 ulp.

Также было сохранено 100000 сэмплов сравнения с референсом в узком диапазоне для
визуализации:

![Alt text](tests/graphs/lalogf.png?raw=true)

## Векторизация

Тривиально можно получить параллелизм по данным, заменив все операции в скалярном
логарифме на соответствующие векторные интринсики AVX512. Такой подход позволяет
считать логарифм 16 скаляров за раз.

Аналогичным образом приходится написать тесты точности, покрывающие тот же интервал,
что и в тестере скалярного логарифма.

Аналогичным образом, выбрав небольшой сабсет точек, получаем графики функции и ее
абсолютной и относительной ошибок:

![Alt text](tests/graphs/lalogf_avx512.png?raw=true)

Видно, что точность сохранилась.

## Тестирование производительности

Методика выбрана схожая с тестированием точности: пройдены все представимые значения
на $[2^{-126}, +\infty)$, но без дополнительных операций и замерено время в циклах
через `rdtsc`. Для стабильности измерения повторялись 8 раз.

Итоговым результатом является для скалярного логарифма является latency,
для векторного - throughput в единицах CPE. Ознакомиться с полной статистикой
измерений можно в [таблице](bench/results.csv). Среднее значение latency 16,
throughput - 2. Таким образом как и ожидалось векторизация дала существенное ускорение
для случая параллельных расчетов.
