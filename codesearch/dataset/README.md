# SPAT
Semantic-and-Naturalness Preserving Auto Transformation

To use this tool, simply type the following command:
```consolo
java -jar SPAT.jar [RuleId] [RootDir] [OutputDir] [PathofJre] \& [PathofotherDependentJar]
```

*[RuleId]* is the transformation rule you want to adopt. 

*[RootDir]* is the root directory path in which you put all your code snippets to be transformed. each ".java'' file is regarded as a code snippet. Each file should contain one Java class. For method-level code snippets, users need to warp each method with a "foo'' class.

*[OutputDir]* is the directory path whre you want to store the transformed code snippets.

*[PathofJre]* is the path of *rt.jar* (usually placed in ".../jre1.x.x\_xxx/lib/''})

*[PathofotherDependentJar]* is optional, one can use it to specify additional dependent libraries.



PS: Due to the version of JAVA, parallel mechanism may not work properly.