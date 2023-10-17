default: model.png model_kvcache.png

model.png: model.puml
	plantuml model.puml -tpng -DPLANTUML_LIMIT_SIZE=20000

model_kvcache.png: model_kvcache.puml
	plantuml model_kvcache.puml -tpng -DPLANTUML_LIMIT_SIZE=40000

model.pdf: model.puml
	plantuml model.puml -tpdf -quiet 2&>1 > /dev/null

model_kvcache.pdf: model_kvcache.puml
	plantuml model_kvcache.puml -tpdf -quiet 2&>1 > /dev/null
