# Решение тестовых заданий

Решение тестовых заданий для лаборатории по AI

[Исходные задания](https://github.com/KernelA/cps-researcher-test)

Решения задач первой части находятся в директории `Part1: Task1, Task3, Task5`, в формате Jupyter notebook.

Решения задач второй части находятся в директории `Part2`.
Архив с предсказаниями находится [здесь](./Part2/raw-data/private.jsonlines.gz)

# Дополнительный анализ статьи

Генеративно-состязательные сети используются достаточно успешно в различных задачах. Применительно к тестовым заданиям их можно использовать в 4 задании при генерации изображений или фигур. В статье [Multi-objective training of Generative Adversarial Networks with multiple
discriminators](https://arxiv.org/pdf/1901.08680.pdf) рассматривается применение методов многокритериальной оптимизации к обучению таких сетей с несколькими классификаторами и одновременной минимизацией потерь. Показано, что такой подход приводит к более качественному генератору.

Всё более актуальна становится проблема проектировании различных систем, где необходимо учитывать множество факторов. Если рассматривать задачи оптимизации как способ поиска наиболее оптимальных параметров, то приходится комбинировать различные критерии, которые отражают степень влияния факторов, в один. Это позволяет решать задачу оптимизации методами однокритериальной оптимизации. Соответственно, упрощает задачу, но здесь осуществляется переход от многокритериальной к однокритериальной постановке задачи. Это может сказаться, в общем случае, на свойствах решения. 

Увеличение интереса к методам многокритериальной оптимизации и развитие численных методов для решения таких задач позволяет по-новому взглянуть на существующие решения различных задач. Например, как к обучению нейронных сетей, где необходимо минимизировать


