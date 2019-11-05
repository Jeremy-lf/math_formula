# math_formula
math symbol recognition

model : Densenet encode model

To pursue better performance, we did four experiments.we also add data_aug on the source code.

first: we resize the wrong label and data (load pre_model)( data_aug)( version_4/  model_2 )              train_4.py

second: we ignore the font of the formula  ( train from zero )( data_aug) ( version_3/  model )           train_3.py

third: we just  want to try load pre_model and the right data without data_aug (load pre_model) no data_aug)
(version_2)                                                                                               train_2.py
