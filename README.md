# IDAO_2021


## Решение первого этапа IDAO 2021


<p>Сначала для решения проблемы классового дисбаланса осуществлялась генерация дополнительных 
изображений для классов с небольшим количеством изображний путём отзеркаливаний и поворотов. </p>

<b>  Simple Aug.ipynb </b>

<p> Далее следовал предпроцессинг данных и построение моделей. Первноаначально небольшой препроцессинг(кроп центра и ресайзинг к 110px на 110px). 
Использовались и другие, а также поиск наиболее ярчайшей области и кроп оттуда, 
но было решено оставить только кроп центра с ресайзингом. </p>

<p> Далее строятся две CNN с разницей в выходном последнем слое. 
Для понижения lr используется scheduler. Сохраняется лучшая модель. 
Далее, в случае энегрии, округляем её к ближайшему классу и сохраняем предсказание. </p>

<b> solution.ipynb </b>

##  <p> Скор первого трека: </p>
 
<p> 390.13 - private score [15058 items], score = auc - mae </p>
<p> public: 0.999226950355 0.00732356857523  </p>
<p> private: 0.98443193283 0.594302032142   </p>


## Решение второго этапа IDAO 2021

<p> В виду того, что некоторое время был недоступен tensorflow, было принятно решение взять бейзлайн на Pytorch 
и добавить к нему свою архитекуру нейросети, предпроцессинг для изображений и параметры.</p>

<p> То есть модели для двух треков обучались независимо, поэтому обучение для второго этапа находится в папке track2.</p>

## <p> Скор второго этапа: </p>

<p> 218.89 - private score [14908 items], score = auc - mae  </p>
<p> public: 0.999957333333 0.0326666666667 </p>
<p> private: 0.970230421163 0.751341561578 </p>

<p> Команда: Made As Described Earlier  </p>
<p> Лидерборд: https://idao.world/results/ </p>
